""" Callbacks related to training progress. """

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
      Value of the history to monitor or callback that determines whether
      this epoch should to a checkpoint. The callback takes the network
      instance as parameter.

      In case ``monitor`` is set to ``None``, the callback will save
      the network at every epoch.

      **Note:** If you supply a lambda expression as monitor, you cannot
      pickle the wrapper anymore as lambdas cannot be pickled. You can
      mitigate this problem by using importable functions instead.
    """
    def __init__(
            self,
            target='model.pt',
            monitor='valid_loss_best',
    ):
        self.monitor = monitor
        self.target = target

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
            if net.verbose > 0:
                print("Checkpoint! Saving model to {}.".format(target))
            net.save_params(target)


class EarlyStopping(Callback):
    """Stop training early if a specified monitor metric did not improve in a
    given amount of epochs.

    Parameters
    ----------
    monitor : str (default='valid_loss_best')
      Value of the history to monitor to decide whether to stop training or not.
      The value is expected to be boolean and is commonly provided by scoring
      callbacks such as :class:`skorch.callbacks.EpochScoring`.

    stop_threshold : int (default=5)
      Number of epochs to wait for improvement of the monitor value until
      the training process is stopped.
    """
    def __init__(self, monitor='valid_loss_best', stop_threshold=5):
        self.monitor = monitor
        self.stop_threshold = stop_threshold
        self.misses_ = 0

    # pylint: disable=arguments-differ
    def on_train_begin(self, net, **kwargs):
        self.misses_ = 0

    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1, self.monitor]:
            self.misses_ += 1
        else:
            self.misses_ = 0
        if self.misses_ == self.stop_threshold:
            if net.verbose:
                print("Stopping since {} did not improve for the last "
                      "{} epochs.".format(self.monitor, self.stop_threshold))
            raise KeyboardInterrupt
