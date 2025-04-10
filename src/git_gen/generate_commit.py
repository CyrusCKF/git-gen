import subprocess

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    pipeline,
)
from huggingface_hub import snapshot_download

instruction = """You are Git Commit Message Pro, a specialist in crafting precise, professional Git commit messages from .diff files. Your role is to analyze these files, interpret the changes, and generate a clear, direct commit message.

Guidelines:
1. Be specific about the type of change (e.g., "Rename variable X to Y", "Extract method Z from class W").
2. Prefer to write it on why and how instead of what changed.
3. Interpret the changes; do not transcribe the diff.
4. If you cannot read the entire file, attempt to generate a message based on the available information."""


def generate_commit():
    # snapshot_download("Qwen/Qwen2.5-Coder-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    result = subprocess.run(["git", "diff"], capture_output=True, text=True)
    result.check_returncode()
    git_diff = result.stdout
    conversation = [
        {"role": "user", "content": instruction + "\n\nInputs:\n" + git_diff},
    ]

    pipe = pipeline("text-generation", model=model, device_map="auto", tokenizer=tokenizer)  # type: ignore
    outputs = pipe(conversation, return_full_text=False)
    return outputs
