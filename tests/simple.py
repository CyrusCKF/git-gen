import subprocess
from pathlib import Path

from git_gen.engine import (load_model_and_tokenizer, make_generate_args,
                            make_prompt, resolve_model_options)


def main():
    size, quantize = resolve_model_options(None, None)
    model, tokenizer = load_model_and_tokenizer(size, quantize, "auto")
    git_diff = _get_git_diff(Path("."))
    conversation = make_prompt(git_diff)
    prompt_kwargs = make_generate_args(conversation, tokenizer, model.device)

    results = model.generate(**prompt_kwargs, max_new_tokens=1024)
    print(tokenizer.decode(results[0]))


def _get_git_diff(folder: Path):
    result = subprocess.run(
        ["git", "diff", "HEAD"], capture_output=True, text=True, cwd=folder
    )
    result.check_returncode()
    return result.stdout


if __name__ == "__main__":
    main()
