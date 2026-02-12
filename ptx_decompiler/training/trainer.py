"""Training loop with curriculum, mixed precision, WandB logging, tqdm progress bars."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ptx_decompiler.training.curriculum import get_max_tier_for_epoch
from ptx_decompiler.training.metrics import exact_match_accuracy, compute_tree_edit_distance
from ptx_decompiler.training.scheduler import get_cosine_schedule_with_warmup


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = None,
        pad_id_ast: int = 0,
        eos_id_ast: int = 2,
        label_smoothing: float = 0.1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        curriculum_sampler: Optional[Any] = None,
        log_interval: int = 50,
        eval_interval: int = 500,
        save_dir: Optional[Path] = None,
        use_wandb: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_id_ast = pad_id_ast
        self.eos_id_ast = eos_id_ast
        self.label_smoothing = label_smoothing
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.curriculum_sampler = curriculum_sampler
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = Path(save_dir) if save_dir else None
        self.use_wandb = use_wandb
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id_ast, label_smoothing=label_smoothing)
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        total_em = 0.0
        total_ted = 0.0
        n_samples = 0

        if self.curriculum_sampler is not None:
            self.curriculum_sampler.set_epoch(epoch)

        max_tier = get_max_tier_for_epoch(epoch)
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch} [tier<=={max_tier}]",
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        for batch_idx, batch in pbar:
            ptx_ids = batch["ptx_ids"].to(self.device)
            ast_input = batch["ast_input_ids"].to(self.device)
            ast_target = batch["ast_target_ids"].to(self.device)
            ptx_mask = batch["ptx_mask"]
            ptx_padding = (~ptx_mask).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits, _ = self.model(ptx_ids, ast_input, ptx_padding)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        ast_target.reshape(-1),
                    )
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(ptx_ids, ast_input, ptx_padding)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    ast_target.reshape(-1),
                )
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                em = exact_match_accuracy(pred, ast_target, self.pad_id_ast, self.eos_id_ast)
                ted = compute_tree_edit_distance(pred, ast_target, self.pad_id_ast, self.eos_id_ast)
                total_em += em * ptx_ids.size(0)
                total_ted += ted * ptx_ids.size(0)
                n_samples += ptx_ids.size(0)

            avg_loss = total_loss / n_batches
            avg_em = total_em / max(n_samples, 1)
            avg_ted = total_ted / max(n_samples, 1)
            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{avg_loss:.4f}", em=f"{avg_em:.4f}", ted=f"{avg_ted:.4f}", lr=f"{lr:.2e}")

            if self.use_wandb and (batch_idx + 1) % self.log_interval == 0:
                try:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/exact_match": avg_em,
                        "train/tree_edit_sim": avg_ted,
                        "train/lr": lr,
                        "step": epoch * len(self.train_loader) + batch_idx,
                    })
                except Exception:
                    pass

        avg_loss = total_loss / max(n_batches, 1)
        avg_em = total_em / max(n_samples, 1)
        avg_ted = total_ted / max(n_samples, 1)
        return {"loss": avg_loss, "exact_match": avg_em, "tree_edit_sim": avg_ted}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_em = 0.0
        total_ted = 0.0
        n_samples = 0
        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        for batch in pbar:
            ptx_ids = batch["ptx_ids"].to(self.device)
            ast_input = batch["ast_input_ids"].to(self.device)
            ast_target = batch["ast_target_ids"].to(self.device)
            ptx_mask = batch["ptx_mask"]
            ptx_padding = (~ptx_mask).to(self.device)

            logits, _ = self.model(ptx_ids, ast_input, ptx_padding)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                ast_target.reshape(-1),
            )
            total_loss += loss.item() * ptx_ids.size(0)
            pred = logits.argmax(dim=-1)
            total_em += exact_match_accuracy(pred, ast_target, self.pad_id_ast, self.eos_id_ast) * ptx_ids.size(0)
            total_ted += compute_tree_edit_distance(pred, ast_target, self.pad_id_ast, self.eos_id_ast) * ptx_ids.size(0)
            n_samples += ptx_ids.size(0)

            n = max(n_samples, 1)
            pbar.set_postfix(
                val_loss=f"{total_loss/n:.4f}",
                val_em=f"{total_em/n:.4f}",
            )

        n = max(n_samples, 1)
        return {
            "val_loss": total_loss / n,
            "val_exact_match": total_em / n,
            "val_tree_edit_sim": total_ted / n,
        }

    def train(self, num_epochs: int) -> None:
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        best_val_em = 0.0
        for epoch in epoch_pbar:
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = time.time() - t0
            best_val_em = max(best_val_em, val_metrics["val_exact_match"])

            epoch_pbar.set_postfix(
                loss=f"{train_metrics['loss']:.4f}",
                val_em=f"{val_metrics['val_exact_match']:.4f}",
                best=f"{best_val_em:.4f}",
                time=f"{elapsed:.0f}s",
            )

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({**train_metrics, **val_metrics, "epoch": epoch})
                except Exception:
                    pass
            if self.save_dir and (epoch + 1) % 5 == 0:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": self.model.state_dict(), "epoch": epoch},
                    self.save_dir / f"checkpoint_epoch_{epoch}.pt",
                )
