import subprocess

import torch
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.pipelines import pipeline

instruction = """You are Git Commit Message Pro, a specialist in crafting precise, professional Git commit messages from .diff files. Your role is to analyze these files, interpret the changes, and generate a clear, direct commit message.

Guidelines:
1. Be specific about the type of change (e.g., "Rename variable X to Y", "Extract method Z from class W").
2. Prefer to write it on why and how instead of what changed.
3. Interpret the changes; do not transcribe the diff.
4. If you cannot read the entire file, attempt to generate a message based on the available information."""


def generate_commit():

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "CyrusCheungkf/git-commit-7B-q4b", quantization_config=quantization_config
    )

    result = subprocess.run(["git", "diff"], capture_output=True, text=True)
    result.check_returncode()
    git_diff = result.stdout
    conversation = [
        {"role": "user", "content": instruction + "\n\nInputs:\n" + git_diff},
    ]

    # tokens = (
    #     torch.tensor(tokenizer.apply_chat_template(conversation)).unsqueeze(0).cuda()
    # )
    # next_token = None
    # end_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    # while next_token != end_id:
    #     with torch.no_grad():
    #         logits = model(tokens).logits
    #         next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    #     tokens = torch.cat((tokens, next_token), dim=1)
    #     print(tokenizer.decode(next_token.item()), end="")
    pipe = pipeline(
        "text-generation", model=model, device_map="auto", tokenizer=tokenizer
    )
    outputs = pipe(conversation, num_return_sequences=4, return_full_text=False)
    return outputs
