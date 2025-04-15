from queue import Queue
import subprocess
from threading import Thread

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.generation.streamers import (
    TextStreamer,
    TextIteratorStreamer,
    BaseStreamer,
)

from transformers.generation.configuration_utils import GenerationConfig

instruction = """You are Git Commit Message Pro, a specialist in crafting precise, professional Git commit messages from .diff files. Your role is to analyze these files, interpret the changes, and generate a clear, direct commit message.

Guidelines:
1. Be specific about the type of change (e.g., "Rename variable X to Y", "Extract method Z from class W").
2. Prefer to write it on why and how instead of what changed.
3. Interpret the changes; do not transcribe the diff.
4. If you cannot read the entire file, attempt to generate a message based on the available information."""


class OutputStream(BaseStreamer):
    STOP_SIGNAL = None

    def __init__(self) -> None:
        super().__init__()
        self.output_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens and push them to queue
        """
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.output_queue.put(value)

    def end(self):
        self.output_queue.put(self.STOP_SIGNAL)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.output_queue.get()
        if value == self.STOP_SIGNAL:
            raise StopIteration()
        else:
            return value


def generate_commit():
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    if torch.cuda.is_available():
        print("CUDA detected. Using quantized model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            "CyrusCheungkf/git-commit-7B-q4b",
            quantization_config=quantization_config,
        )
    else:
        print("CUDA not detected. Using a weaker model")
        model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            "CyrusCheungkf/git-commit-1.5B"
        )

    result = subprocess.run(["git", "diff"], capture_output=True, text=True)
    result.check_returncode()
    git_diff = result.stdout
    conversation = [
        {"role": "user", "content": instruction + "\n\nInputs:\n" + git_diff},
    ]

    tokens = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    streamer = OutputStream()
    generation_kwargs = dict(
        inputs=tokens,
        streamer=streamer,
        num_return_sequences=4,
        pad_token_id=tokenizer.eos_token_id,
        max_length=2048,
        # temperature=0.1,
        # top_p=0.8,
        # repetition_penalty=1.25,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for output in streamer:
        print([tokenizer.decode(o) for o in output], flush=True)
