import subprocess
from pathlib import Path
from pprint import pprint

from git_gen.engine import load_model, make_prompt


def main():
    model = load_model()
    git_diff = _get_git_diff(Path("."))
    conversation = make_prompt(git_diff)
    print("Model inputs:")
    for message in conversation:
        print(message["role"].upper(), message.get("content"))
    chat_completer = model.create_chat_completion(conversation, stream=True)
    print("=" * 10)
    print("Model outputs:")
    for output in chat_completer:
        try:
            text = output["choices"][0]["delta"]["content"]  # type: ignore
        except:
            text = None
        if text is not None:
            print(text, end="", flush=True)


def _get_git_diff(folder: Path):
    result = subprocess.run(
        ["git", "diff", "HEAD"], capture_output=True, text=True, cwd=folder
    )
    result.check_returncode()
    return result.stdout


if __name__ == "__main__":
    main()
