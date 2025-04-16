"""Contains helper functions to use the model"""

import logging
import subprocess
from pathlib import Path
from pprint import pprint
from queue import Queue
from threading import Thread
from typing import Iterable

import torch
from rich.console import Console
from transformers.generation.utils import GenerationMixin
from transformers.generation.streamers import BaseStreamer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding

logger = logging.getLogger(__name__)


MODEL_SIZE_QUANT_TO_IDS = {
    ("3B", False): "CyrusCheungkf/git-commit-3B",
    ("3B", True): "CyrusCheungkf/git-commit-3B-q8b",
    ("7B", False): "CyrusCheungkf/git-commit-7B",
    ("7B", True): "CyrusCheungkf/git-commit-7B-q4b",
}


def resolve_model_options(model_size: str | None, quantize_model: bool | None):
    """Set the default value for model options dynamically, and validate them.

    Refer to `MODEL_SIZE_QUANT_TO_IDS` to choose the model ID

    Returns:
        Tuple of (actual_size, actual_is_quantize)
    """
    if quantize_model is not None:
        if not torch.cuda.is_available() and quantize_model:
            raise ValueError(
                "'quantize_model' is set to True but CUDA is not available."
                "Please set 'quantize_model' to False."
            )
        is_quantized = quantize_model
    else:
        if torch.cuda.is_available():
            logger.info("CUDA detected. Using quantized model")
            is_quantized = True
        else:
            logger.info("CUDA not detected. Using non-quantized model")
            is_quantized = False

    if model_size is not None:
        size = model_size
    else:
        # default size to 7B. May choose depending on user's machine
        size = "7B"
    if size in ("3B", "1.5B"):
        raise NotImplementedError(f"model_size {size} is not supported yet.")

    return size, is_quantized


def load_model_and_tokenizer(
    model_size: str, quantize_model: bool, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Return a tuple of (model, tokenizer)"""
    model_id = MODEL_SIZE_QUANT_TO_IDS[(model_size, quantize_model)]
    logging.info(f"Loading model {model_id}")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id, low_cpu_mem_usage=quantize_model, device_map=device
    )
    assert isinstance(model, PreTrainedModel)

    # only use fine-tuned qwen2.5 model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    return model, tokenizer


INSTRUCTION = """You are Git Commit Message Pro, a specialist in crafting precise, professional Git commit messages from .diff files. Your role is to analyze these files, interpret the changes, and generate a clear, direct commit message.

Guidelines:
1. Be specific about the type of change (e.g., "Rename variable X to Y", "Extract method Z from class W").
2. Prefer to write it on why and how instead of what changed.
3. Interpret the changes; do not transcribe the diff.
4. If you cannot read the entire file, attempt to generate a message based on the available information."""


def make_prompt(input: str) -> list[dict[str, str]]:
    """Return a suitable chat input with instruction for tokenizer."""
    conversation = [
        {"role": "user", "content": INSTRUCTION + "\n\nInputs:\n" + input},
    ]
    return conversation


def make_generate_args(
    conversation: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> dict:
    """Process conversation prompt to be arguments of `model.generate`"""
    tokens = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    assert isinstance(tokens, BatchEncoding)
    tokens = tokens.to(device)
    return {
        "inputs": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "pad_token_id": tokenizer.eos_token_id,
    }


class OutputStream(BaseStreamer):
    """Stream the raw model output"""

    STOP_SIGNAL = None

    def __init__(self) -> None:
        super().__init__()
        self.output_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        """Receive tokens and push them to queue"""
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.output_queue.put(value)

    def end(self):
        self.output_queue.put(self.STOP_SIGNAL)

    def __iter__(self):
        return self

    def __next__(self):
        """Wait for next output in a blocking manner until the end."""
        value = self.output_queue.get()
        if value == self.STOP_SIGNAL:
            raise StopIteration()
        else:
            return value


def stream_generation(model: GenerationMixin, generate_kwargs: dict) -> Iterable[int]:
    """Return generator of token ids one at a time.

    Args:
        generate_kwargs: Arguments for `model.generate`
    """
    streamer = OutputStream()
    generate_kwargs = generate_kwargs.copy()
    generate_kwargs["streamer"] = streamer
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    for output in streamer:
        yield output.tolist()
