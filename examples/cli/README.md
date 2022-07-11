# skorch helpers for command line interfaces (CLIs)

Often you want to wrap up your experiments by writing a small script
that allows others to reproduce your work. With the help of skorch and
the fire library, it becomes very easy to write command line
interfaces without boilerplate. All arguments pertaining to skorch or
its PyTorch module are immediately available as command line
arguments, without the need to write a custom parser. If docstrings in
the numpydoc specification are available, there is also an
comprehensive help for the user. Overall, this allows you to make your
work reproducible without the usual hassle.

This example is a showcase of how easy CLIs become with skorch.

## Installation

To use this functionaliy, you need some further libraries that are not
part of skorch, namely fire and numpydoc. You can install them thusly:

```bash
python -m pip install fire numpydoc
```

## Usage

The `train.py` file contains an example of how to write your own CLI
with the help of skorch. As you can see, this file almost exclusively
consists of the proper logic, there is no argument parsing
involved.

When you write your own script, only the following bits need to be
added:

```python

import fire
from skorch.helper import parse_args

# your model definition and data fetching code below
...

def main(**kwargs):
    X, y = get_data()
    my_model = get_model()

    # important: wrap the model with the parsed arguments
    parsed = parse_args(kwargs)
    my_model = parsed(my_model)

    my_model.fit(X, y)


if __name__ == '__main__':
    fire.Fire(main)

```

Note: The function you pass to `fire.Fire` shouldn't have any
positional arguments, otherwise the displayed help will not work
correctly; this is a quirk of fire.

This even works if your neural net is part of an sklearn pipeline, in
which case the help extends to all other estimators of your pipeline.

In case you would like to change some defaults for the net (e.g. using
a `batch_size` of 256 instead of 128), this is also possible. You
should have a dictionary containing your new defaults and pass it as
an additional argument to `parse_args`:

```python

my_defaults = {'batch_size': 128, 'module__hidden_units': 30}

def main(**kwargs):
    ...
    parsed = parse_args(kwargs, defaults=my_defaults)
    my_model = parsed(my_model)

```

This will update the displayed help to your new defaults, as well as
set the parameters on the net or pipeline for you. However, the
arguments passed via the commandline have precedence. Thus, if you
additionally pass ``--batch_size 512`` to the script, batch size will
be 512.

For more information on how to use fire, follow [this
link](https://github.com/google/python-fire).

## Restrictions

Almost all arguments should work out of the box. Therefore, you get
command line arguments for the number of epochs, learning rate, batch
size, etc. for free. Moreover, you can access the module parameters
with the double-underscore notation as usual with skorch
(e.g. `--module__num_units 100`). This should cover almost all common
cases.

Parsing command line arguments that are non-primitive Python objects
is more difficult, though. skorch's custom parsing should support
normal Python types and simple custom objects, e.g. this works:
`--module__nonlin 'torch.nn.RReLU(0.1, upper=0.4)'`. More complex
parsing might not work. E.g., it is currently not possible to add new
callbacks through the command line (but you can modify existing ones
as usual).

## Running the script

### Getting Help

In this example, there are two variants, only the net ("net") and the
net within an sklearn pipeline ("pipeline"). To get general help for
each, run:

```bash
python train.py net -- --help
python train.py pipeline -- --help
```

To get help for model-specific parameters, run:

```bash
python train.py net --help
python train.py pipeline --help
```

### Training a Model

Run

```bash
python train.py net  # only the net
python train.py pipeline  # net with pipeline
```

with the defaults.

Example with just the net and some non-defaults:

```bash
python train.py net --n_samples 1000 --output_file 'model.pkl' --lr 0.1 --max_epochs 5 --device 'cuda' --module__hidden_units 50 --module__nonlin 'torch.nn.RReLU(0.1, upper=0.4)' --callbacks__valid_acc__on_train --callbacks__valid_acc__name train_acc
```

Example with an sklearn pipeline:

```bash
python train.py pipeline --n_samples 1000 --net__lr 0.1 --net__module__nonlin 'torch.nn.LeakyReLU()' --scale__minmax__feature_range '(-2, 2)' --scale__normalize__norm l1
```
