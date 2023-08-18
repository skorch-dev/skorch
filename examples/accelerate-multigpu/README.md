# Testing skorch with accelerate in multi GPU setting

This directory contains a couple of script used to test skorch with accelerate in a multi-GPU setting. The scripts cannot run as unit tests because they require a specific hardware setup not provided by the GitHub Action runners.

## `run-with-skorch.py`

The full history of this can be found here: https://github.com/skorch-dev/skorch/issues/944

There was an issue with using skorch in a multi-GPU setting with accelerate. After some searching, it turns out there were two problems:

1. skorch did not call `accelerator.gather_for_metrics`, which resulted in `y_pred` not having the correct size. For more on this, consult the [accelerate docs](https://huggingface.co/docs/accelerate/quicktour#distributed-evaluation).
2. accelerate has an issue with beeing deepcopied, which happens for instance when using `GridSearchCV`. The problem is that some references get messed up, resulting in the `GradientState` of the `accelerator` instance and of the `dataloader` to diverge. Therefore, the `accelerator` did not "know" when the last batch was encountered and was thus unable to remove the dummy samples added for multi-GPU inference.

The fix for 1. is provided in the same PR as this was added. For 2., the problem has been fixed [in accelerate](https://github.com/huggingface/accelerate/pull/1694) and is contained in the 0.21 release.

This example contains two scripts, one involving skorch and one with skorch completely removed. The scripts reproduce the issue in a multi-GPU setup (tested on a GCP VM instance with two T4's). Unfortunately, the GitHub Action runners don't have such an option, which is why there is no unit test being added for the bug.

Run the scripts like this:

```sh
accelerate launch <script.py>
```

The accelerate config is:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## `run-save-load.py`

The context of this script is that there were issues with saving and loading when using `AccelerateMixin`. The provided script is to ensure that everything works as expected. Same as the first one, for a proper test, this script needs to run in a multi-GPU setting. For more information, check PR #1008.

Run the scripts like this:

```sh
accelerate launch run-save-load.py
```

The accelerate config is the same.
