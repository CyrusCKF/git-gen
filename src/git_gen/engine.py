"""Contains helper functions to use the model"""

import logging

from llama_cpp import (ChatCompletionRequestMessage,
                       ChatCompletionRequestUserMessage, Llama)

logger = logging.getLogger(__name__)


def load_model() -> Llama:
    """Return the pretrained gguf model"""
    model_id = "CyrusCheungkf/git-commit-7B"
    logging.info(f"Loading model {model_id}")
    model = Llama.from_pretrained(
        repo_id=model_id, filename="*.gguf", verbose=False, n_ctx=32768
    )
    return model


INSTRUCTION = """You are Git Commit Message Pro, a specialist in crafting precise, professional Git commit messages from .diff files. Your role is to analyze these files, interpret the changes, and generate a clear, direct commit message.

Guidelines:
1. Be specific about the type of change (e.g., "Rename variable X to Y", "Extract method Z from class W").
2. Prefer to write it on why and how instead of what changed.
3. Interpret the changes; do not transcribe the diff.
4. If you cannot read the entire file, attempt to generate a message based on the available information.
5. Be concise and summarize the most important changes. Keep your response in 1 sentence."""


def make_prompt(input: str) -> list[ChatCompletionRequestMessage]:
    """Return a suitable chat input with instruction"""
    conversation: list[ChatCompletionRequestMessage] = [
        ChatCompletionRequestUserMessage(
            role="user", content=INSTRUCTION + "\n\nInputs:\n" + input
        ),
    ]
    return conversation
