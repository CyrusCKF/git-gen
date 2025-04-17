# git-gen

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/git-gen.svg)](https://pypi.org/project/git-gen)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/git-gen.svg)](https://pypi.org/project/git-gen) -->

Generate git commit message with a local LLM, which has been fine-tuned on 300K high-quality Python commits and is able to analyze the purposes of the changes

## Installation

Required Python 3.10+  
To install this package, run:

```shell
pip install git-gen-utils
```

## Uses

To generate git commit message in current working directory, run:

```shell
git-gen
```

Generate with custom config:

```shell
git-gen --temperature 0.2 --top_k 50 --top_p 0.7
```

To get all command options, run:

```shell
git-gen -h
```

which should output

```plaintext
Usage:  [OPTIONS]

Options:
  --path PATH                 Path to the project, which should contain a '.git' folder. Default to current working directory.
  --max_tokens INTEGER RANGE  The maximum numbers of tokens to generate.  [default: 1024; x>=1]
  --temperature FLOAT RANGE   Control the overall probabilities of the generation. Prefer lower temperature for higher accuracy and higher temperature for
                              more varied outputs.  [default: 0.7; x>0]
  --top_k INTEGER RANGE       Limit the number of vocabs to consider.  [default: 20; x>=1]
  --top_p FLOAT RANGE         Limit the set of vocabs by cumulative probability.  [default: 0.8; 0<x<=1]
  -h, --help                  Show this message and exit.
```

## FAQ

There are a lot of similar projects. Why would I use this?

> A good commit message should reflect the purpose of the changes made concisely. Most LLMs failed to do that at zero-shot learning. This project aims to generate messages that focus on the "why"  instead of "what", by using a fine-tuned model on 300K high-quality Python commits.
>
> Moreover, it only uses local LLM. You code is safe from the internet.

## License

`git-gen` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Notes

### Build and push project

```shell
python -m pip install --upgrade build
python -m build
python -m pip install --upgrade twine
python -m twine upload --repository pypi dist/* --verbose
```
