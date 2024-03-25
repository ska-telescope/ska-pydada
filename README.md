# SKA PyDADA

[![Documentation Status](https://readthedocs.org/projects/ska-telescope-ska-pydada/badge/?version=latest)](https://developer.skao.int/projects/ska-pydada/en/latest/)

`ska-pydada` is a Python library that can be used to read DADA files used in pulsar timing. The documentation for this
project can be found at [SKA developer portal](https://developer.skao.int/projects/ska-pydada/en/latest/)

## Developer Setup

### Required Packages

The following packages are required for local development of this project. If the developer is not using Ubuntu
or Window Subsystem for Linux (WSL) then they will need to find an equivalent package.

* [graphviz](https://graphviz.org/)
* [pandoc](https://pandoc.org/)

```bash
sudo apt-get update
sudo apt-get install pandoc graphviz
```

### Poetry Setup

No matter what environment that you use, you will need to make sure that Poetry is
installed and that you have the Poetry shell running.

Install Poetry based on [Poetry Docs](https://python-poetry.org/docs/). Ensure that you're using at least 1.3.2, as the
`pyproject.toml` and `poetry.lock` files have been migrated to the Poetry 1.3.2.  The following command will install Poetry
version 1.3.2

    curl -sSL https://install.python-poetry.org | python3 - --version 1.3.2

After having Poetry installed, run the following command to be able to install the project. This will create a virtual env for you before starting.

    poetry install

If this is successful you should be able to use your favourite editor/IDE to develop in this project.

To activate the poetry environment then run in the same directory:

    poetry shell

(For VS Code, you can then set your Python Interpreter to the path of this virtual env.)

### Ensure Linting Before Commit

It is highly recommended that linting is performed **before** committing your code.  This project
has a `pre-commit` hook that can be enabled.  SKA Make machinery provides the following command
that can be used by developers to enable the lint check pre-commit hook.

    make dev-git-hooks

After this has been applied, `git commit` commands will run the pre-commit hook. If you
want to avoid doing that for some work in progress (WIP) then run the following command
instead

    git commit --no-verify <other params>

### Editor Configuration

This project has an `.editorconfig` file that can be used with IDEs/editors that support
[EditorConfig](https://editorconfig.org/).  Both VS Code and Vim have plugins for this,
please check your favourite editor for use of the plugin.

For those not familiar with EditorConfig, it uses a simple configuration file that
instructs your editor to do some basic formatting, like tabs as 4 spaces for Python or
leaving tabs as tabs for Makefiles, or even trimming trailing whitespace of lines.


## Download the source code

First, clone the repository and all of its submodules

    git clone --recursive git@gitlab.com:ska-telescope/pst/ska-pydada.git

Next, change to the newly cloned directory

    cd ska-pydada
    poetry shell
    poetry install

### Building

### Documentation Build

API documentation for the library is generated with Doxygen, which is then converted into ReadTheDocs format by Sphinx, Breathe and Exhale. The documentation is built via 

    make docs-build html

### Python Linting Build

This project requires that the code is well formated and linted by `pylint` and `mypy`.

Your code can be formated by running:

    make python-format

While the code can be linted by running:

    make python-lint

To ensure that formatting happens before linting, add the following to `PrivateRules.mak`

```make
python-pre-lint: python-format

.PHONY: python-pre-lint
```

## License

See the LICENSE file for details.
