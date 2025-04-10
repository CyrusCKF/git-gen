from pathlib import Path
import click


def echo(**kwargs):
    types = {k: type(v) for k, v in kwargs.items()}
    click.echo(f"Received args {kwargs}. Types: {types}")
