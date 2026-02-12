"""Gradio demo: PTX input -> AST + CUDA output, attention heatmap, compilation badge."""

import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from ptx_decompiler.data import normalize_ptx, parse_sexp
from ptx_decompiler.data.renderer import CUDARenderer
from ptx_decompiler.inference import DecompilationPipeline, CompilationVerifier
from ptx_decompiler.visualization import ast_to_graphviz_svg


def load_demo_artifacts():
    """Load tokenizers and model if available; else return None for placeholder mode."""
    try:
        from ptx_decompiler.tokenizer import PTXTokenizer, ASTTokenizer
        import pandas as pd
        model_path = ROOT / "checkpoints" / "checkpoint_epoch_29.pt"
        data_path = ROOT / "dataset_100k.parquet"
        if not model_path.exists() or not data_path.exists():
            return None, None, None
        df = pd.read_parquet(data_path)
        ptx_tok = PTXTokenizer(max_vocab_size=2000)
        ptx_tok.build_vocab(df["ptx_normalized"].tolist())
        ast_tok = ASTTokenizer()
        ptx_to_ast = torch.full((len(ptx_tok),), -1, dtype=torch.long)
        for t, pid in ptx_tok.vocab.items():
            if t in ast_tok.vocab:
                ptx_to_ast[pid] = ast_tok.vocab[t]
        from ptx_decompiler.model import PTXDecompilerModel
        model = PTXDecompilerModel(
            ptx_vocab_size=len(ptx_tok),
            ast_vocab_size=len(ast_tok),
            ptx_to_ast_map=ptx_to_ast,
        )
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        pipeline = DecompilationPipeline(model, ptx_tok, ast_tok, device=torch.device("cpu"))
        return pipeline, ptx_tok, ast_tok
    except Exception:
        return None, None, None


PIPELINE, _, _ = load_demo_artifacts()


def run_decompile(ptx_input: str) -> tuple:
    if not ptx_input.strip():
        return "", "", "", "⚠ No input", ""
    if PIPELINE is None:
        ast_sexp = "(ADD A B)"
        try:
            tree = parse_sexp(ast_sexp)
            cuda = CUDARenderer().kernel_source(tree)
        except Exception:
            cuda = "// Load model and tokenizers to run decompilation"
        return (
            ast_sexp,
            cuda,
            "⚠ Model not loaded (place checkpoint and dataset in project root)",
            "❓",
            "",
        )
    result = PIPELINE.decompile(ptx_input)
    status = "✅ Compiles" if result.compiles else "❌ Does not compile"
    svg = ast_to_graphviz_svg(result.ast) if result.ast else None
    html_svg = f'<div style="overflow:auto">{svg}</div>' if svg and not svg.startswith("<p>") else (svg or "")
    return result.ast, result.cuda, status, status, html_svg


with gr.Blocks(title="DeepPTX: Neural PTX Decompiler", theme=gr.themes.Soft()) as app:
    gr.Markdown("# DeepPTX — PTX to CUDA Decompiler")
    with gr.Row():
        with gr.Column():
            ptx_in = gr.Textbox(
                label="PTX input",
                placeholder="Paste normalized PTX instructions here...",
                lines=8,
            )
            btn = gr.Button("Decompile")
        with gr.Column():
            ast_out = gr.Textbox(label="Generated AST (S-expression)", lines=6)
            cuda_out = gr.Textbox(label="Decompiled CUDA", lines=10)
    with gr.Row():
        status = gr.Markdown("Status: —")
        comp_badge = gr.Markdown("—")
    with gr.Row():
        ast_svg = gr.HTML(label="AST tree")
    btn.click(
        fn=run_decompile,
        inputs=[ptx_in],
        outputs=[ast_out, cuda_out, status, comp_badge, ast_svg],
    )
    gr.Examples(
        examples=[
            ["add.f32 %f0 %f1 %f2"],
        ],
        inputs=ptx_in,
        label="Examples",
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
