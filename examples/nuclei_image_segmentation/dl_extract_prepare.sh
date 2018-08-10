#!/usr/bin/env sh

if [ ! -f "data/stage1_train.zip" ]; then
	printf "Downloading data from kaggle\n"
	kaggle competitions download -c data-science-bowl-2018 -f stage1_train.zip -p data
else
	printf "data/stage1_train.zip already exists\n"
fi

if [ ! -d "data/cells" ]; then
	printf "Unzipping datasets/stage1_train.zip to data/cells\n"
	mkdir -p data/cells
	unzip data/stage1_train.zip -d data/cells
else
	printf "data/cells already exists\n"
fi

printf "Preparing data\n"
python prepare_dataset.py
