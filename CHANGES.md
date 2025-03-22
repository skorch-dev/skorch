# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Add an example of hyper-parameter optimization using [Optuna](https://optuna.org/) [here](https://github.com/skorch-dev/skorch/tree/master/examples/optuna) (#1098)
### Added

- Add Contributing Guidelines for skorch. (#1097)
### Changed

- Loading of skorch nets using pickle: When unpickling a skorch net, you may come across a PyTorch warning that goes: "FutureWarning: You are using torch.load with weights_only=False [...]"; to avoid this warning, pickle the net again and use the new pickle file (#1092)

### Fixed

## [1.1.0]

### Added

- Added a [notebook](https://github.com/skorch-dev/skorch/blob/master/notebooks/Learning_Rate_Scheduler.ipynb) that shows how to use Learning Rate Scheduler in skorch.(#1074)

### Changed

- All neural net classes now inherit from sklearn's [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html). This is to support compatibility with sklearn 1.6.0 and above. Classification models additionally inherit from [`ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html) and regressors from [`RegressorMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html). (#1078)
- When using the `ReduceLROnPlateau` learning rate scheduler, we now record the learning rate in the net history (`net.history[:, 'event_lr']` by default). It is now also possible to to step per batch, not only by epoch (#1075)
- The learning rate scheduler `.simulate()` method now supports adding step args which is useful when simulation policies such as `ReduceLROnPlateau` which expect metrics to base their schedule on. (#1077)
- Removed deprecated `skorch.callbacks.scoring.cache_net_infer` (#1088)

### Fixed

- Fix an issue with using `NeuralNetBinaryClassifier` with `torch.compile` (#1058)

## [1.0.0] - 2024-05-27

The 1.0.0 release of skorch is here. We think that skorch is at a very stable point, which is why a 1.0.0 release is appropriate. There are no plans to add any breaking changes or major revisions in the future. Instead, our focus now is to keep skorch up-to-date with the latest versions of PyTorch and scikit-learn, and to fix any bugs that may arise.

## [0.15.0] - 2023-09-04

### Added
- Add the option to globally override the use of caching in scoring callbacks on the net by setting the `use_caching` argument on the net (this overrides the settings of individual callbacks) (#971)
- Add support for saving and loading parameters with [safetensors](https://github.com/huggingface/safetensors/); use `net.save_params(..., use_safetensors=True)` and `net.load_params(..., use_safetensors=True)` (requires to install the `safetensors` library) (#970)

### Changed
- Nets pickled with skorch version 0.11 can no longer be loaded in version 0.15 (see #880); to transition these nets, pickle them in a skorch version between 0.12 and 0.14, then load them in 0.15

### Fixed

- Fixed a couple of issues when saving and loading parameters while using accelerate (via `AccelerateMixin`) in a multi-GPU setting, and some other minor accelerate issues (#1008, #1009)
- Installing skorch with the `[testing]` option now installs all dev requirements (#1015)

## [0.14.0] - 2023-06-24

### Added

- Add version logging to `NeptuneLogger` callback (#964)
- Add support for [zero-shot and few-shot classification](https://skorch.readthedocs.io/en/latest/user/llm.html#using-language-models-as-zero-and-few-shot-classifiers) with the help of Large Language Models and the Hugging Face transformers library

### Changed
- Moved from `pkg_resources` to `importlib` and subsequently dropping support for Python 3.7
  as PyTorch moved dropped support and the version itself hit EOL (#928 and #983)

- `NeuralNetRegressor` can now be fitted with 1-dimensional `y`, which is necessary in some specific circumstances (e.g. in conjunction with sklearn's `BaggingRegressor`, see #972); for this to work correctly, the output of the of the PyTorch module should also be 1-dimensional; the existing default, i.e. having `y` and `y_pred` be 2-dimensional, remains the recommended way of using `NeuralNetRegressor`

### Fixed

## [0.13.0] - 2023-05-17

### Added
- Add support for compiled PyTorch modules using the `torch.compile` function, introduced in [PyTorch 2.0 release](https://pytorch.org/get-started/pytorch-2.0/), which can greatly improve performance on new GPU architectures; to use it, initialize your net with the `compile=True` argument, further compilation arguments can be specified using the dunder notation, e.g. `compile__dynamic=True`
- Add a class [`DistributedHistory`](https://skorch.readthedocs.io/en/latest/history.html#skorch.history.DistributedHistory) which should be used when training in a multi GPU setting (#955)
- `SkorchDoctor`: A helper class that assists in understanding and debugging the neural net training, see [this notebook](https://nbviewer.org/github/skorch-dev/skorch/blob/master/notebooks/Skorch_Doctor.ipynb) (#912)
- When using `AccelerateMixin`, it is now possible to prevent unwrapping of the modules by setting `unwrap_after_train=True` (#963)

### Changed

### Fixed
- Fixed install command to work with recent changes in Google Colab (#928)
- Fixed a couple of bugs related to using non-default modules and criteria (#927)
- Fixed a bug when using `AccelerateMixin` in a multi-GPU setup (#947)
- `_get_param_names` returns a list instead of a generator so that subsequent
  error messages return useful information instead of a generator `repr`
  string (#925)
- Fixed a bug that caused modules to not be sufficiently unwrapped at the end of training when using `AccelerateMixin`, which could prevent them from being pickleable (#963)

## [0.12.1] - 2022-11-18

### Changed

- `NeptuneLogger` was updated to work with recent versions of Neptune client (v0.14.3 or higher); it now logs some additional data, including the model summary, configuration, and learning rate (when available) (#906)

### Fixed

- Fixed an error that could occur with specific combinations of gpytorch and PyTorch versions (#913)

## [0.12.0] - 2022-10-07

### Added
- Added `load_best` attribute to `EarlyStopping` callback to automatically load module weights of the best result at the end of training
- Added a method, `trim_for_prediction`, on the net classes, which trims the net from everything not required for using it for prediction; call this after fitting to reduce the size of the net
- Added experimental support for [huggingface accelerate](https://github.com/huggingface/accelerate); use the provided mixin class to add advanced training capabilities provided by the accelerate library to skorch
- Add integration for Huggingface tokenizers; use `skorch.hf.HuggingfaceTokenizer` to train a Huggingface tokenizer on your custom data; use `skorch.hf.HuggingfacePretrainedTokenizer` to load a pre-trained Huggingface tokenizer
- Added support for creating model checkpoints on Hugging Face Hub using [`HfHubStorage`](https://skorch.readthedocs.io/en/latest/hf.html#skorch.hf.HfHubStorage)
- Added a [notebook](https://nbviewer.org/github/skorch-dev/skorch/blob/master/notebooks/CORA-geometric.ipynb) that shows how to use skorch with PyTorch Geometric (#863)

### Changed
- The minimum required scikit-learn version has been bumped to 0.22.0
- Initialize data loaders for training and validation dataset once per fit call instead of once per epoch ([migration guide](https://skorch.readthedocs.io/en/stable/user/FAQ.html#migration-from-0-11-to-0-12))
- It is now possible to call `np.asarray` with `SliceDataset`s (#858)

### Fixed
- Fixed a bug in `SliceDataset` that prevented it to be used with `to_numpy` (#858)
- Fixed a bug that occurred when loading a net that has device set to None (#876)
- Fixed a bug that in some cases could prevent loading a net that was trained with CUDA without CUDA
- Enable skorch to work on M1/M2 Apple MacBooks (#884)

## [0.11.0] - 2021-10-11

### Added

- Added `load_best` attribute to `Checkpoint` callback to automatically load state of the best result at the end of training
- Added a `get_all_learnable_params` method to retrieve the named parameters of all PyTorch modules defined on the net, including of criteria if applicable
- Added `MlflowLogger` callback for logging to Mlflow (#769)
- Added `InputShapeSetter` callback for automatically setting the input dimension of the PyTorch module
- Added a new module to support Gaussian Processes through [GPyTorch](https://gpytorch.ai/). To learn more about it, read the [GP documentation](https://skorch.readthedocs.io/en/latest/user/probabilistic.html) or take a look at the [GP notebook](https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Gaussian_Processes.ipynb). This feature is experimental, i.e. the API could be changed in the future in a backwards incompatible way (#782)

### Changed

- Changed the signature of `validation_step`, `train_step_single`, `train_step`, `evaluation_step`, `on_batch_begin`, and `on_batch_end` such that instead of receiving `X` and `y`, they receive the whole batch; this makes it easier to deal with datasets that don't strictly return an `(X, y)` tuple, which is true for quite a few PyTorch datasets; please refer to the [migration guide](https://skorch.readthedocs.io/en/latest/user/FAQ.html#migration-from-0-10-to-0-11) if you encounter problems (#699)
- Checking of arguments to `NeuralNet` is now during `.initialize()`, not during `__init__`, to avoid raising false positives for yet unknown module or optimizer attributes
- Modules, criteria, and optimizers that are added to a net by the user are now first class: skorch takes care of setting train/eval mode, moving to the indicated device, and updating all learnable parameters during training (check the [docs](https://skorch.readthedocs.io/en/latest/user/customization.html#initialization-and-custom-modules) for more details, #751)
- `CVSplit` is renamed to `ValidSplit` to avoid confusion (#752)

### Fixed

- Fixed a few bugs in the `net.history` implementation (#776)
- Fixed a bug in `TrainEndCheckpoint` that prevented it from being unpickled (#773)

## [0.10.0] - 2021-03-23

### Added

- Added `SacredLogger` callback for logging to Sacred (#725)
- CLI helper function now also supports normal (i.e. non-skorch) sklearn estimators
- Disabling all callbacks is now supported (which allows reducing overhead,
  which is especially relevant for small models).
- `LRScheduler` now correctly passes the value being monitored to `ReduceLROnPlateau`. (#738)

### Changed

- We no longer pass the `epoch` parameter to LR schedulers, since that parameter has been deprecated. We now rely on the scheduler to keep track of the epoch itself.
- Changed implementation of `net.history` access to make it faster; this should result in a nice speedup when dealing with very small model/data but otherwise not have any noticeable effects; if you encounter bugs, though, please create an issue

### Fixed

## [0.9.0] - 2020-08-30

### Added

- Added the `event_name` argument for `LRScheduler` for optional recording of LR changes inside `net.history`. NOTE: Supported only in Pytorch>=1.4
- Make it easier to add custom modules or optimizers to a neural net class by automatically registering them where necessary and by making them available to set_params
- Added the `step_every` argument for `LRScheduler` to set whether the scheduler step should be taken on every epoch or on every batch.
- Added the `scoring` module with `loss_scoring` function, which computes the net's loss (using `get_loss`) on provided input data.
- Added a parameter `predict_nonlinearity` to `NeuralNet` which allows users to control the nonlinearity to be applied to the module output when calling `predict` and `predict_proba` (#637, #661)
- Added the possibility to save the criterion with `save_params` and with checkpoint callbacks
- Added the possibility to save custom modules with `save_params` and with checkpoint callbacks

### Changed

- Removed support for schedulers with a `batch_step()` method in `LRScheduler`.
- Raise `FutureWarning` in `CVSplit` when `random_state` is not used. Will raise an exception in a future (#620)
- The behavior of method `net.get_params` changed to make it more consistent with sklearn: it will no longer return "learned" attributes like `module_`; therefore, functions like `sklearn.base.clone`, when called with a fitted net, will no longer return a fitted net but instead an uninitialized net; if you want a copy of a fitted net, use `copy.deepcopy` instead;`net.get_params` is used under the hood by many sklearn functions and classes, such as `GridSearchCV`, whose behavior may thus be affected by the change. (#521, #527)
- Raise `FutureWarning` when using `CyclicLR` scheduler, because the default behavior has changed from taking a step every batch to taking a step every epoch. (#626)
- Set train/validation on criterion if it's a PyTorch module (#621)
- Don't pass `y=None` to `NeuralNet.train_split` to enable the direct use of split functions without positional `y` in their signatures. This is useful when working with unsupervised data (#605).
- `to_numpy` is now able to unpack dicts and lists/tuples (#657, #658)
- When using `CrossEntropyLoss`, softmax is now automatically applied to the output when calling `predict` or `predict_proba`

### Fixed

- Fixed a bug where `CyclicLR` scheduler would update during both training and validation rather than just during training.
- Fixed a bug introduced by moving the `optimizer.zero_grad()` call outside of the train step function, making it incompatible with LBFGS and other optimizers that call the train step several times per batch (#636)
- Fixed pickling of the `ProgressBar` callback (#656)

## [0.8.0] - 2020-04-11

### Added

- Added `NeptuneLogger` callback for logging experiment metadata to neptune.ai (#586)
- Add `DataFrameTransformer`, an sklearn compatible transformer that helps working with pandas DataFrames by transforming the DataFrame into a representation that works well with neural networks (#507)
- Added `WandbLogger` callback for logging to Weights & Biases (#607)
- Added `None` option to `device` which leaves the device(s) unmodified (#600)
- Add `PassthroughScoring`, a scoring callback that just calculates the average score of a metric determined at batch level and then writes it to the epoch level (#595)

### Changed

- When using caching in scoring callbacks, no longer uselessly iterate over the data; this can save time if iteration is slow (#552, #557)
- Cleaned up duplicate code in the `fit_loop` (#564)

### Future Changes

- WARNING: In release 0.10.0 of skorch, Python 3.5 support will be officially dropped (#634)

### Fixed

- Make skorch compatible with sklearn 0.22 (#571, #573, #575)
- Fixed a bug that could occur when a new "settable" (via `set_params`) attribute was added to `NeuralNet` whose name starts the same as an existing attribute's name (#590)

## [0.7.0] - 2019-11-29

### Added

- More careful check for wrong parameter names being passed to `NeuralNet` (#500)
- More helpful error messages when trying to predict using an uninitialized model
- Add `TensorBoard` callback for automatic logging to tensorboard
- Make `NeuralNetBinaryClassifier` work with `sklearn.calibration.CalibratedClassifierCV`
- Improve `NeuralNetBinaryClassifier` compatibility with certain sklearn metrics (#515)
- `NeuralNetBinaryClassifier` automatically squeezes module output if necessary (#515)
- `NeuralNetClassifier` now has a `classes_` attribute after fit is called, which is inferred from y by default (#465, #486)
- `NeuralNet.load_params` with a checkpoint now initializes when needed (#497)

### Changed

- Improve numerical stability when using `NLLLoss` in `NeuralNetClassifer` (#491)
- Refactor code to make gradient accumulation easier to implement (#506)
- `NeuralNetBinaryClassifier.predict_proba` now returns a 2-dim array; to access the "old" `y_proba`, take `y_proba[:, 1]` (#515)
- `net.history` is now a property that accesses `net.history_`, which stores the `History` object (#527)
- Remove deprecated `skorch.callbacks.CyclicLR`, use `torch.optim.lr_scheduler.CyclicLR` instead

### Future Changes

- WARNING: In a future release, the behavior of method `net.get_params` will change to make it more consistent with sklearn: it will no longer return "learned" attributes like `module_`. Therefore, functions like `sklearn.base.clone`, when called with a fitted net, will no longer return a fitted net but instead an uninitialized net. If you want a copy of a fitted net, use `copy.deepcopy` instead. Note that `net.get_params` is used under the hood by many sklearn functions and classes, such as `GridSearchCV`, whose behavior may thus be affected by the change. (#521, #527)

### Fixed

- Fixed a bug that caused `LoadInitState` not to work with `TrainEndCheckpoint` (#528)
- Fixed `NeuralNetBinaryClassifier` wrongly squeezing the batch dimension when using `batch_size = 1` (#558)


## [0.6.0] - 2019-07-19

### Added

- Adds FAQ entry regarding the initialization behavior of `NeuralNet` when passed instantiated models. (#409)
- Added CUDA pickle test including an artifact that supports testing on CUDA-less CI machines
- Adds `train_batch_count` and `valid_batch_count` to history in training loop. (#445)
- Adds score method for NeuralNetClassifier, NeuralNetBinaryClassifier, and NeuralNetRegressor (#469)
- Wrapper class for torch Datasets to make them work with some sklearn features (e.g. grid search). (#443)

### Changed

- Repository moved to https://github.com/skorch-dev/skorch/, please change your git remotes
- Treat cuda dependent attributes as prefix to cover values set using `set_params` since
  previously `"criterion_"` would not match `net.criterion__weight` as set by
  `net.set_params(criterion__weight=w)`
- skorch pickle format changed in order to improve CUDA compatibility, if you have pickled models, please re-pickle them to be able to load them in the future
- `net.criterion_` and its parameters are now moved to target device when using criteria that inherit from `torch.nn.Module`. Previously the user had to make sure that parameters such as class weight are on the compute device
- skorch now assumes PyTorch >= 1.1.0. This mainly affects learning rate schedulers, whose inner workings have been changed with version 1.1.0. This update will also invalidate pickled skorch models after a change introduced in PyTorch optimizers.

### Fixed

- Include requirements in MANIFEST.in
- Add `criterion_` to `NeuralNet.cuda_dependent_attributes_` to avoid issues with criterion
  weight tensors from, e.g., `NLLLoss` (#426)
- `TrainEndCheckpoint` can be cloned by `sklearn.base.clone`. (#459)


## [0.5.0] - 2018-12-13

### Added

- [Basic usage notebook][1810251445] now runs on Google Colab
- [Advanced usage notebook][1810261633] now runs on Google Colab
- [MNIST with scikit-learn and skorch][1811011230] now runs on Google Colab
- Better user-facing messages when module or optimizer are re-initialized
- Added an experimental API (`net._register_virtual_param`) to register "virtual"
  parameters on the network with custom setter functions. (#369)
- Setting parameters `lr`, `momentum`, `optimizer__lr`, etc. no longer resets
  the optmizer. As of now you can do `net.set_params(lr=0.03)` or
  `net.set_params(optimizer__param_group__0__momentum=0.86)` without triggering
  a re-initialization of the optimizer (#369)
- Support for scipy sparse CSR matrices as input (as, e.g., returned by sklearn's
  `CountVectorizer`); note that they are cast to dense matrices during batching
- Helper functions to build command line interfaces with almost no
  boilerplate, [example][1811191713] that shows usage

[1810251445]: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Basic_Usage.ipynb
[1810261633]: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb
[1811011230]: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb
[1811191713]: https://github.com/skorch-dev/skorch/tree/master/examples/cli

### Changed

- Reduce overhead of `BatchScoring` when using `train_loss_score` or `valid_loss_score` by skipping superfluous inference step (#381)
- The `on_grad_computed` callback function will yield an iterable for `named_parameters` only when it is used to reduce the run-time overhead of the call (#379)
- Default `fn_prefix` in `TrainEndCheckpoint` is now `train_end_` (#391)
- Issues a warning when `Checkpoints`'s `monitor` parameter is set to `monitor` and the history contains `<monitor>_best`. (#399)

### Fixed

- Re-initialize optimizer when `set_params` is called with `lr` argument (#372)
- Copying a `SliceDict` now returns a `SliceDict` instead of a `dict` (#388)
- Calling `==` on `SliceDict`s now works as expected when values are numpy arrays and torch tensors


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

[1]: https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/examples/nuclei_image_segmentation/Nuclei_Image_Segmentation.ipynb
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


[Unreleased]: https://github.com/skorch-dev/skorch/compare/v0.10.0...HEAD
[0.4.0]: https://github.com/skorch-dev/skorch/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/skorch-dev/skorch/compare/v0.4.0...v0.5.0
[0.6.0]: https://github.com/skorch-dev/skorch/compare/v0.5.0...v0.6.0
[0.7.0]: https://github.com/skorch-dev/skorch/compare/v0.6.0...v0.7.0
[0.8.0]: https://github.com/skorch-dev/skorch/compare/v0.7.0...v0.8.0
[0.9.0]: https://github.com/skorch-dev/skorch/compare/v0.8.0...v0.9.0
[0.10.0]: https://github.com/skorch-dev/skorch/compare/v0.9.0...v0.10.0
[0.11.0]: https://github.com/skorch-dev/skorch/compare/v0.10.0...v0.11.0
[0.12.0]: https://github.com/skorch-dev/skorch/compare/v0.11.0...v0.12.0
[0.12.1]: https://github.com/skorch-dev/skorch/compare/v0.12.0...v0.12.1
[0.13.0]: https://github.com/skorch-dev/skorch/compare/v0.12.1...v0.13.0
[0.14.0]: https://github.com/skorch-dev/skorch/compare/v0.13.0...v0.14.0
[0.15.0]: https://github.com/skorch-dev/skorch/compare/v0.14.0...v0.15.0
[1.0.0]: https://github.com/skorch-dev/skorch/compare/v0.15.0...v1.0.0
[1.1.0]: https://github.com/skorch-dev/skorch/compare/v1.0.0...v1.1.0
