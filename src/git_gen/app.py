import logging
import pprint
from pathlib import Path
import subprocess

import click
from rich.logging import RichHandler
from transformers.generation.configuration_utils import GenerationConfig
from git_gen.engine import (
    load_model_and_tokenizer,
    make_generate_args,
    make_prompt,
    resolve_model_options,
    stream_generation,
)
from rich.console import Console

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


@click.command(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "show_default": True,
        "max_content_width": 160,
    }
)
@click.option(
    "--model_size",
    type=click.Choice(["3B", "7B"], case_sensitive=False),
    help="Size of model",
)
@click.option(
    "--quantize",
    "quantize_model",
    is_flag=False,
    flag_value=True,
    type=bool,
    help="Whether to use quantized model or not. NOTE: this is only supported on CUDA platform",
)
@click.option(
    "--device",
    default="auto",
    help='Device to run on. Common values include: "cpu", "cuda", "mps"',
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
    "--num_msg",
    "num_return_sequences",
    default=4,
    show_default=True,
    help="Number of generated messages. This will slightly increase computations needed.",
)
@click.option(
    "--max_new_tokens",
    type=click.IntRange(min=1),
    help="The maximum numbers of tokens to generate.",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0, min_open=True),
    help="Control the overall probabilities of the generation. Prefer lower temperature"
    " for higher accuracy and higher temperature for more varied outputs.",
)
@click.option(
    "--top_k",
    type=click.IntRange(min=1),
    help="Limit the number of vocabs to consider.",
)
@click.option(
    "--top_p",
    type=click.FloatRange(0, 1, min_open=True),
    help="Limit the set of vocabs by cumulative probability.",
)
def app(
    model_size: str | None,
    quantize_model: bool | None,
    device: str,
    project_folder: Path,
    **kwargs,
):
    """Learn more about generation config here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig"""
    console = Console()

    size, quantize = resolve_model_options(model_size, quantize_model)
    model, tokenizer = load_model_and_tokenizer(size, quantize, device)
    try:
        git_diff = _get_git_diff(project_folder)
    except:
        logger.error(
            f'Error occurs when running "git diff" on {project_folder}.'
            " Make sure this is a valid git repo and git is installed in your PATH"
        )
        return

    conversation = make_prompt(git_diff)
    prompt_kwargs = make_generate_args(conversation, tokenizer, model.device)

    if model.generation_config is not None:
        final_gen_params = _get_final_gen_params(model.generation_config, kwargs)
        console.log("Generating with params:", final_gen_params)
    generate_kwargs = prompt_kwargs | kwargs
    token_generator = stream_generation(model, generate_kwargs)
    for tokens in token_generator:
        click.echo(tokens)


def _get_git_diff(folder: Path):
    result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=folder)
    result.check_returncode()
    return result.stdout


def _get_final_gen_params(from_model: GenerationConfig, from_inputs: dict):
    model_gen_dict = from_model.to_dict()
    final_config = {}
    for k, v in from_inputs.items():
        final_config[k] = model_gen_dict[k] if v is None else v
    return final_config


if __name__ == "__main__":
    app()
