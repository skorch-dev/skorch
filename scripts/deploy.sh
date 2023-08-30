#!/usr/bin/env bash
set -eu
set -o pipefail

PYTORCH_VERSION=${PYTORCH_VERSION:-""}
PYTHON_VERSION="3.9"
TWINE_VERSION="\>3,\<4.0.0dev" # escaped <,> are necessary for conda run
CONDA_ENV="skorch-deploy"

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

# make sure that conda is up-to-date
conda update -n base -c defaults -q -y conda

# Remove previous deploy environment
set +e
conda env remove -y -n $CONDA_ENV
set -e

echo "creating empty conda env"
conda create -y -q -n $CONDA_ENV python=$PYTHON_VERSION

remove_env() {
    conda env remove -q -y -n $CONDA_ENV
    if [ -d build ]; then
        rm -rf build
    fi
    if [ -d dist ]; then
        rm -rf dist
    fi
}

run_in_env() {
    # shellcheck disable=SC2068
    conda run -n "$CONDA_ENV" --no-capture-output $@
}

trap remove_env EXIT

echo "installing dependencies"
conda install -c pytorch -y "pytorch==${PYTORCH_VERSION}"
run_in_env python -m pip install "twine${TWINE_VERSION}"
# Workaround for error `AttributeError: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'`
# due to outdated system pyOpenSSL - see also: https://askubuntu.com/q/1428181
run_in_env python -m pip install pyOpenSSL --upgrade
run_in_env python -m pip install -r requirements.txt
run_in_env python -m pip install -r requirements-dev.txt
run_in_env python -m pip install .
run_in_env python -m pip list

run_in_env pytest -x

# check if README can be rendered correctly on PyPI
run_in_env python -m pip install readme-renderer
run_in_env python -m readme_renderer README.rst > /dev/null

run_in_env python setup.py sdist bdist_wheel

if [[ $1 == "live" ]]; then
    run_in_env twine upload dist/*"$(cat VERSION)"*
else
    run_in_env twine upload --repository-url https://test.pypi.org/legacy/ dist/*"$(cat VERSION)"*
fi
