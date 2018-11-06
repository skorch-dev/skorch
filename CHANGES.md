# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [Basic usage notebook][1810251445] now runs on Google Colab
- [Advanced usage notebook][1810261633] now runs on Google Colab
- [MNIST with scikit-learn and skorch][1811011230] now runs on Google Colab
- Better user-facing messages when module or optimizer are re-initialized
- Reduce overhead of `BatchScoring` when using `train_loss_score` or `valid_loss_score` by skipping superfluous inference step (#381)


[1810251445]: https://colab.research.google.com/github/dnouri/skorch/blob/master/notebooks/Basic_Usage.ipynb
[1810261633]: https://colab.research.google.com/github/dnouri/skorch/blob/master/notebooks/Advanced_Usage.ipynb
[1811011230]: https://colab.research.google.com/github/dnouri/skorch/blob/master/notebooks/MNIST.ipynb

### Fixed

- Re-initialize optimizer when `set_params` is called with `lr` argument (#372)

## [0.4.0] - 2018-10-24

### Added

- Support for PyTorch 0.4.1
- There is no need to explicitly name callbacks anymore (names are assigned automatically, name conflicts are resolved).
- You can now access the training data in the `on_grad_computed` event
- There is a new [image segmentation example][1]
- Easily create toy network instances for quick experiments using [`skorch.toy.make_classifier`][2] and friends
- New [`ParamMapper`][3] callback to modify/freeze/unfreeze parameters at certain point in time during training:
```python
>>> from sklearn.callbacks import Freezer, Unfreezer
>>> net = Net(module, callbacks=[Freezer('layer*.weight'), Unfreezer('layer*.weight', at=10)])
```
- Refactored `EpochScoring` for easier sub-classing
- `Checkpoint` callback now supports saving the optimizer, this avoids problems with stateful
  optimizers such as `Adam` or `RMSprop` (#360)
- Added `LoadInitState` callback for easy continued training from checkpoints (#360)
- `NeuralNetwork.load_params` now supports loading from `Checkpoint` instances
- Added documentation for [saving and loading][4]

[1]: https://nbviewer.jupyter.org/github/dnouri/skorch/blob/master/examples/nuclei_image_segmentation/Nuclei_Image_Segmentation.ipynb
[2]: https://skorch.readthedocs.io/en/latest/toy.html
[3]: https://skorch.readthedocs.io/en/latest/callbacks.html#skorch.callbacks.ParamMapper
[4]: https://skorch.readthedocs.io/en/latest/user/save_load.html

### Changed

- The `ProgressBar` callback now determines the batches per epoch automatically by default (`batches_per_epoch=auto`)
- The `on_grad_computed` event now has access to the current training data batch

### Deprecated

- Deprecated `filtered_optimizer` in favor of `Freezer` callback (#346)
- `NeuralNet.load_params` and `NeuralNet.save_params` deprecate `f` parameter for the sake
  of `f_optimizer`, `f_params` and `f_history` (#360)

### Fixed

- `uses_placeholder_y` should not require existence of `y` field (#311)
- LR scheduler creates `batch_idx` on first run (#314)
- Use `OrderedDict` for callbacks to fix python 3.5 compatibility issues (#331)
- Make `to_tensor` work correctly with `PackedSequence` (#335)
- Rewrite `History` to not use any recursion to avoid memory leaks during exceptions (#312)
- Use `flaky` in some neural network tests to hide platform differences
- Fixes ReduceLROnPlateau when mode == max (#363)
- Fix disconnected weights between net and optimizer after copying the net with `copy.deepcopy` (#318)
- Fix a bug that intefered with loading CUDA models when the model was a CUDA tensor but
  the net was configured to use the CPU (#354, #358)


[Unreleased]: https://github.com/dnouri/skorch/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/dnouri/skorch/compare/v0.3.0...v0.4.0
