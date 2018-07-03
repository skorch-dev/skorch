""" Callbacks related to training progress. """

import numpy as np
from skorch.callbacks import Callback
from skorch.exceptions import SkorchException


__all__ = ['Checkpoint', 'EarlyStopping']


class Checkpoint(Callback):
    """Save the model during training if the given metric improved.

    This callback works by default in conjunction with the validation
    scoring callback since it creates a ``valid_loss_best`` value
    in the history which the callback uses to determine if this
    epoch is save-worthy.

    You can also specify your own metric to monitor or supply a
    callback that dynamically evaluates whether the model should
    be saved in this epoch.

    Example:

    >>> net = MyNet(callbacks=[Checkpoint()])
    >>> net.fit(X, y)

    Example using a custom monitor where only models are saved in
    epochs where the validation *and* the train loss is best:

    >>> monitor = lambda net: all(net.history[-1, (
    ...     'train_loss_best', 'valid_loss_best')])
    >>> net = MyNet(callbacks=[Checkpoint(monitor=monitor)])
    >>> net.fit(X, y)

    Parameters
    ----------
    target : file-like object, str
      File path to the file or file-like object.
      See NeuralNet.save_params for details what this value may be.

      If the value is a string you can also use format specifiers
      to, for example, indicate the current epoch. Accessible format
      values are ``net``, ``last_epoch`` and ``last_batch``.
      Example to include last epoch number in file name:

      >>> cb = Checkpoint(target="target_{last_epoch[epoch]}.pt")

    monitor : str, function, None
      Value of the history to monitor or callback that determines
      whether this epoch should lead to a checkpoint. The callback
      takes the network instance as parameter.

      In case ``monitor`` is set to ``None``, the callback will save
      the network at every epoch.

      **Note:** If you supply a lambda expression as monitor, you cannot
      pickle the wrapper anymore as lambdas cannot be pickled. You can
      mitigate this problem by using importable functions instead.

    sink : callable (default=print)
      The target that the information about created checkpoints is
      sent to. By default, the output is printed to stdout, but the
      sink could also be a logger or :func:`~skorch.utils.noop`.
    """
    def __init__(
            self,
            target='model.pt',
            monitor='valid_loss_best',
            sink=print,
    ):
        self.monitor = monitor
        self.target = target
        self.sink = sink

    def on_epoch_end(self, net, **kwargs):
        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(net)
        else:
            try:
                do_checkpoint = net.history[-1, self.monitor]
            except KeyError as e:
                raise SkorchException(
                    "Monitor value '{}' cannot be found in history. "
                    "Make sure you have validation data if you use "
                    "validation scores for checkpointing.".format(e.args[0]))

        if do_checkpoint:
            target = self.target
            if isinstance(self.target, str):
                target = self.target.format(
                    net=net,
                    last_epoch=net.history[-1],
                    last_batch=net.history[-1, 'batches', -1],
                )
            self._sink("Checkpoint! Saving model to {}.".format(target), net.verbose)
            net.save_params(target)

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)


class EarlyStopping(Callback):
    """Callback for stopping training when scores don't improve.

    Stop training early if a specified `monitor` metric did not
    improve in `patience` number of epochs by at least `threshold`.

    Parameters
    ----------
    monitor : str (default='valid_loss')
      Value of the history to monitor to decide whether to stop
      training or not.  The value is expected to be double and is
      commonly provided by scoring callbacks such as
      :class:`skorch.callbacks.EpochScoring`.

    lower_is_better : bool (default=True)
      Whether lower scores should be considered better or worse.

    patience : int (default=5)
      Number of epochs to wait for improvement of the monitor value
      until the training process is stopped.

    threshold : int (default=1e-4)
      Ignore score improvements smaller than `threshold`.

    threshold_mode : str (default='rel')
        One of `rel`, `abs`. Decides whether the `threshold` value is
        interpreted in absolute terms or as a fraction of the best
        score so far (relative)

    sink : callable (default=print)
      The target that the information about created checkpoints is
      sent to. By default, the output is printed to stdout, but the
      sink could also be a logger or :func:`~skorch.utils.noop`.

    """
    def __init__(
            self,
            monitor='valid_loss',
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
            sink=print,
    ):
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.misses_ = 0
        self.dynamic_threshold_ = None
        self.sink = sink

    # pylint: disable=arguments-differ
    def on_train_begin(self, net, **kwargs):
        if self.threshold_mode not in ['rel', 'abs']:
            raise ValueError("Invalid threshold mode: '{}'"
                             .format(self.threshold_mode))
        self.misses_ = 0
        self.dynamic_threshold_ = np.inf if self.lower_is_better else -np.inf

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
        if self.misses_ == self.patience:
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)
            raise KeyboardInterrupt

    def _is_score_improved(self, score):
        if self.lower_is_better:
            return score < self.dynamic_threshold_
        return score > self.dynamic_threshold_

    def _calc_new_threshold(self, score):
        """Determine threshold based on score."""
        if self.threshold_mode == 'rel':
            abs_threshold_change = self.threshold * score
        else:
            abs_threshold_change = self.threshold

        if self.lower_is_better:
            new_threshold = score - abs_threshold_change
        else:
            new_threshold = score + abs_threshold_change
        return new_threshold

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)
