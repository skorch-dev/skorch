
Mostly adapted from [here](https://github.com/pytorch/examples/tree/master/word_language_model).

### Training:

	python train.py

This will do a grid search on a fraction of the PTB dataset and store
checkpoints of the model in a file (default is `./model.pt`).

### Generation:

	python generate.py

The default setting is that during training in `train.py` a checkpoint model
is written to `./model.pt` by default. This model checkpoint is also used
for generation in `generate.py`. The generated text is written to 
`generated.txt` by default.

