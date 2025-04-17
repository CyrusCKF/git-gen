import logging
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from git_gen.engine import load_model, make_prompt
from git_gen.views import stream_lines

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
console = Console()


@click.command(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "show_default": True,
        "max_content_width": 160,
    }
)
@click.option(
    "--path",
    "project_folder",
    type=click.Path(path_type=Path),
    default=".",
    show_default=False,
    help="Path to the project, which should contain a '.git' folder."
    " Default to current working directory.",
)
@click.option(
    "--max_tokens",
    type=click.IntRange(min=1),
    default=1024,
    help="The maximum numbers of tokens to generate.",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0, min_open=True),
    default=0.7,
    help="Control the overall probabilities of the generation. Prefer lower temperature"
    " for higher accuracy and higher temperature for more varied outputs.",
)
@click.option(
    "--top_k",
    type=click.IntRange(min=1),
    default=20,
    help="Limit the number of vocabs to consider.",
)
@click.option(
    "--top_p",
    type=click.FloatRange(0, 1, min_open=True),
    default=0.8,
    help="Limit the set of vocabs by cumulative probability.",
)
def app(
    project_folder: Path,
    **kwargs,
):
    model = load_model()
    try:
        git_diff = _get_git_diff(project_folder)
    except:
        logger.error(
            f'Error occurs when running "git diff" on {project_folder}.'
            " Make sure this is a valid git repo and git is installed in your PATH"
        )
        return

    conversation = make_prompt(git_diff)
    chat_completer = model.create_chat_completion(conversation, stream=True, **kwargs)
    # chat_completer = [{"choices": [{"delta": {"content": "Output sentence" * 4}}]}]

    # Stream outputs to console
    title = (
        "Generating messages (This may take up to a few minutes if the diff is large)"
    )
    with stream_lines(console, title) as line_streamer:
        for output in chat_completer:
            try:
                text = output["choices"][0]["delta"]["content"]  # type: ignore
            except:
                text = None
            if text is not None:
                line_streamer.append([text])
        lines = line_streamer.lines
    message = lines[0]
    console.print()

    # Repeatedly ask and validate for user input
    table = Table.grid(padding=(1, 0), expand=True)
    table.add_row("Would you like to commit with the message?", style="i")
    table.add_row("    " + message)
    table.add_row("Type y to commit, n to exit", style="#696969")
    console.print(Panel(table, expand=False))

    def validate_input(inpt):
        try:
            return inpt == "y" or inpt == "n"
        except:
            return False

    user_input = console.input(">> ")
    while not validate_input(user_input):
        logger.error("Invalid input. Please try again")
        user_input = console.input(">> ")

    if user_input == "n":
        console.log("Bye bye")
    else:
        _git_commit_all(project_folder, message)
        console.log("Git committed successfully")


def _get_git_diff(folder: Path):
    result = subprocess.run(
        ["git", "diff", "HEAD"], capture_output=True, text=True, cwd=folder
    )
    result.check_returncode()
    return result.stdout


def _git_commit_all(folder: Path, message: str):
    subprocess.run(["git", "add", "-A"], cwd=folder)
    subprocess.run(["git", "commit", "-m", message], cwd=folder)


if __name__ == "__main__":
    app()
