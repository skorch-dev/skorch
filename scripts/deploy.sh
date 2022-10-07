#!/usr/bin/env bash
set -e

PYTORCH_VERSION=${PYTORCH_VERSION:-""}
PYTHON_VERSION="3.9"
TWINE_VERSION=">3,<4.0.0dev"
CONDA_ENV="skorch-deploy"
CONDA_ENV_YML="environment.yml"

if [[ $# -gt 1 ]] || [[ $1 != "live" && $1 != "stage" ]]; then
	echo "Usage $0 [live|stage]" >&2
	exit 1
fi

if [ -z "$PYTORCH_VERSION" ]; then
	echo "Set a PYTORCH_VERSION in the environment!"
	exit 1
fi

# check if worktree is not dirty, see https://stackoverflow.com/a/5737794
test -z "$(git status --porcelain --untracked-files=no)"

conda update -q -y conda

# Remove previous deploy environment
set +e
conda env remove -y -n $CONDA_ENV
set -e

conda env create -q -n $CONDA_ENV -f $CONDA_ENV_YML "python=${PYTHON_VERSION}"

remove_env() {
    source deactivate
    conda env remove -q -y -n $CONDA_ENV
    if [ -d build ]; then
        rm -rf build
    fi
    if [ -d dist ]; then
        rm -rf dist
    fi
}

trap remove_env EXIT

source activate $CONDA_ENV
conda install -q -y "twine==${TWINE_VERSION}"
conda install -c pytorch -y "pytorch==${PYTORCH_VERSION}"
python -m pip install -r requirements-dev.txt
python -m pip install .

pytest -x

# check if README can be rendered correctly on PyPI
python -m pip install readme-renderer
python -m readme_renderer README.rst > /dev/null

python setup.py sdist bdist_wheel

if [[ $1 == "live" ]]; then
    twine upload dist/*$(cat VERSION)*
else
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*$(cat VERSION)*
fi
