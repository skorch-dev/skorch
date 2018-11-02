#!/usr/bin/env bash
set -e

if [[ $# -gt 1 ]] || [[ $1 != "live" && $1 != "stage" ]]; then
	echo "Usage $0 [live|stage]" >&2
	exit 1
fi

conda env create -n skorch-deploy -f environment-deploy.yml

remove_env() {
    source deactivate
    conda env remove -y -n skorch-deploy
    rm -rf build dist
}

trap remove_env EXIT

source activate skorch-deploy

pytest -x

python setup.py sdist bdist_wheel

if [[ $1 == "live" ]]; then
    twine upload dist/*$(cat VERSION)*
else
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*$(cat VERSION)*
fi
