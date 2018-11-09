==================
Saving and Loading
==================

Skorch provides callbacks: :class:`.Checkpoint`, :class:`.TrainEndCheckpoint`,
and :class:`.LoadInitState` to handle saving and loading models during
training. To demonstrate these features, we generate a dataset and create
a simple module:

.. code:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from torch import nn

    from skorch import NeuralNetClassifier

    X, y = make_classification(1000, 10, n_informative=5, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    class MyModule(nn.Sequential):
        def __init__(self, num_units=10):
            super().__init__(
                nn.Linear(10, num_units),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(num_units, 10),
                nn.Linear(10, 2),
                nn.Softmax(dim=-1)
            )

We create a checkpoint setting ``dirname`` to ``'exp1'``. This will
configure the checkpoint to save the model parameters, optimizer,
and history into a directory named ``'exp1'``.

.. code:: python

    # First run

    from skorch.callbacks import Checkpoint, TrainEndCheckpoint

    cp = Checkpoint(dirname='exp1')
    final_cp = TrainEndCheckpoint(dirname='exp1')
    net = NeuralNetClassifier(
        MyModule, lr=0.5, callbacks=[cp, final_cp]
    )

    _ = net.fit(X, y)

    # prints
      epoch    train_loss    valid_acc    valid_loss    cp     dur
    -------  ------------  -----------  ------------  ----  ------
          1        0.6200       0.8209        0.4765     +  0.0232
          2        0.3644       0.8557        0.3474     +  0.0238
          3        0.2875       0.8806        0.3201     +  0.0214
          4        0.2514       0.8905        0.3080     +  0.0237
          5        0.2333       0.9154        0.2844     +  0.0203
          6        0.2177       0.9403        0.2164     +  0.0215
          7        0.2194       0.9403        0.2159     +  0.0220
          8        0.2027       0.9403        0.2299        0.0202
          9        0.1864       0.9254        0.2313        0.0196
         10        0.2024       0.9353        0.2333        0.0221

By default, the checkpoint observes ``valid_loss`` and will save
the model when the ``valid_loss`` is lowest. This can be seen by
the ``+`` mark in the ``cp`` column of the logs.

On our first run, the validation loss did not improve after the 7th
epoch. We can continue training from this checkpoint with a lower
learning rate by using :class:`.LoadInitState`:

.. code:: python

    from skorch.callbacks import LoadInitState

    cp = Checkpoint(dirname='exp1')
    load_state = LoadInitState(cp)
    net = NeuralNetClassifier(
        MyModule, lr=0.1, callbacks=[cp, load_state]
    )

    _ = net.fit(X, y)

    # prints

      epoch    train_loss    valid_acc    valid_loss    cp     dur
    -------  ------------  -----------  ------------  ----  ------
          8        0.1939       0.9055        0.2626     +  0.0238
          9        0.2055       0.9353        0.2031     +  0.0239
         10        0.1992       0.9453        0.2101        0.0182
         11        0.2033       0.9453        0.1947     +  0.0211
         12        0.1825       0.9104        0.2515        0.0185
         13        0.2010       0.9453        0.1927     +  0.0187
         14        0.1508       0.9453        0.1952        0.0198
         15        0.1679       0.9502        0.1905     +  0.0181
         16        0.1516       0.9453        0.1864     +  0.0192
         17        0.1576       0.9453        0.1804     +  0.0184

Since we started from the previous checkpoint which ended at epoch 7,
the second run starts at epoch 8, continuing from the first checkpoint.
With a lower learning rate, the validation loss was able to improve!

Notice in the first run, we included a :class:`.TrainEndCheckpoint` in the
callbacks. This checkpoint saves the model at the end of training. This
checkpoint can be passed to :class:`.LoadInitState` to continue training:

.. code:: python

    cp_from_final = Checkpoint(dirname='exp1', fn_prefix='from_final_')
    load_state = LoadInitState(final_cp)
    net = NeuralNetClassifier(
        MyModule, lr=0.1, callbacks=[cp_from_final, load_state]
    )

    _ = net.fit(X, y)

    # prints

      epoch    train_loss    valid_acc    valid_loss    cp     dur
    -------  ------------  -----------  ------------  ----  ------
         11        0.1663       0.9453        0.2166     +  0.0282
         12        0.1880       0.9403        0.2237        0.0178
         13        0.1813       0.9353        0.1993     +  0.0161
         14        0.1744       0.9353        0.1955     +  0.0150
         15        0.1538       0.9303        0.2053        0.0077
         16        0.1473       0.9403        0.1947     +  0.0078
         17        0.1563       0.9254        0.1989        0.0074
         18        0.1558       0.9403        0.1877     +  0.0075
         19        0.1534       0.9254        0.2318        0.0074
         20        0.1779       0.9453        0.1814     +  0.0074

In this run, training started at epoch 11, continuing from the
end of the first run which ended at epoch 10. We created a new checkpoint
with ``fn_prefix`` set to ``'from_final'`` to prefix the saved filenames
with ``'from_final'`` to make sure this checkpoint does not override the
checkpoint from the previous run.

Since our ``MyModule`` class allows ``num_units`` to be adjusted,
we can start a new experiment by changing the ``dirname``:

.. code:: python

    cp = Checkpoint(dirname='exp2')
    load_state = LoadInitState(cp)
    net = NeuralNetClassifier(
        MyModule, lr=0.5,
        callbacks=[cp, load_state],
        module__num_units=20,
    )

    _ = net.fit(X, y)

    # prints

      epoch    train_loss    valid_acc    valid_loss    cp     dur
    -------  ------------  -----------  ------------  ----  ------
          1        0.5256       0.8856        0.3624     +  0.0181
          2        0.2956       0.8756        0.3416     +  0.0222
          3        0.2280       0.9453        0.2299     +  0.0211
          4        0.1948       0.9303        0.2136     +  0.0232
          5        0.1800       0.9055        0.2696        0.0223
          6        0.1605       0.9403        0.1906     +  0.0190
          7        0.1594       0.9403        0.2027        0.0184
          8        0.1319       0.9303        0.1910        0.0220
          9        0.1558       0.9254        0.1923        0.0189
         10        0.1432       0.9303        0.2219        0.0192

This stores the model into the ``'exp2'`` directory. Since this is the first run,
the :class:`.LoadInitState` callback does not do anything. If we were to run the above
script again, the :class:`.LoadInitState` callback will load the model from the
checkpoint.

In the run above, the last checkpoint was created at epoch 6, we can load this
checkpoint to predict with it:

.. code:: python

    net = NeuralNetClassifier(
        MyModule, lr=0.5, module__num_units=20,
    )
    net.initialize()
    net.load_params(checkpoint=cp)

    y_pred = net.predict(X)

In this case, it is important to initialize the neutral net before running
:meth:`.NeutralNet.load_params`.
