import logging

import click
from rich.logging import RichHandler

from .generate_commit import generate_commit

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)


@click.command()
@click.option("--count", type=int, help="Number of greetings.")
@click.option("--path", type=click.Path(), help="The file path.")
def app(**kwargs):
    messages = generate_commit()
    click.echo(messages)


if __name__ == "__main__":
    app()
