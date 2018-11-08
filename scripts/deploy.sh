#!/usr/bin/env bash
set -e

PYTORCH_VERSION="0.4.1"
PYTHON_VERSION="3.6"
TWINE_VERSION="1.12.1"
CONDA_ENV="skorch-deploy"
CONDA_ENV_YML="environment.yml"
DEV_REQ="requirements-dev.txt"

if [[ $# -gt 1 ]] || [[ $1 != "live" && $1 != "stage" ]]; then
	echo "Usage $0 [live|stage]" >&2
	exit 1
fi

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
conda install -q -y "--file=${DEV_REQ}"
conda install -q -y "twine==${TWINE_VERSION}"
conda install -c pytorch -y "pytorch==${PYTORCH_VERSION}"
python setup.py install

pytest -x

python setup.py sdist bdist_wheel

if [[ $1 == "live" ]]; then
    twine upload dist/*$(cat VERSION)*
else
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*$(cat VERSION)*
fi
