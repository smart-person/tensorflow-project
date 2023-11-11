# Federated Learning Under Time-Varying Communication Constraints and Intermittent Client Availability

This directory contains source code for experiments in "Federated Learning Under Time-Varying 
Communication Constraints and Intermittent ClientAvailability". 

We build on top of the public repository found 
([here](https://github.com/google-research/federated/tree/master/optimization)). 


## Using this directory

This library uses [TensorFlow Federated](https://www.tensorflow.org/federated).

Some pip packages are required by this library, and may need to be installed:

```
pip install absl-py
pip install attr
pip install dm-tree
pip install numpy
pip install pandas
pip install tensorflow
pip install tensorflow-federated
```
### Install Bazel
You will need to install [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide
[here](https://docs.bazel.build/versions/master/install.html) for installation
instructions.

```sh
!apt install apt-transport-https curl gnupg
!curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
!mv bazel.gpg /etc/apt/trusted.gpg.d/
!echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
!apt update && apt install bazel
```

## Directory structure

This directory is broken up into three task directories. Each task directory
contains task-specific libraries (such as libraries for loading the correct
dataset), as well as libraries for performing federated training. These are 
in the `optimization/{task}` folders.

A single binary for running these tasks can be found at
`main/federated_trainer.py`. This binary will, according to `absl` flags, run
any of the six task-specific federated training libraries.

There is also a `shared` directory with utilities specific to these experiments,
such as implementations of metrics used for evaluation.

## Example usage

Suppose we wish to train a convolutional network on EMNIST for purposes of
character recognition (`emnist_cr`), using federated optimization. Various
aspects of the federated training procedure can be customized via `absl` flags.
For example, from this directory one could run:

```
bazel run main:federated_trainer -- \
--task=cifar100 \
--total_rounds=100 \
--client_optimizer=sgd \
--client_learning_rate=0.1 \
--client_batch_size=20 \
--server_optimizer=sgd \
--server_learning_rate=1.0 \
--clients_per_round=10 \
--client_epochs_per_round=1 \
--experiment_name=emnist_fedavg_experiment \
--schedule=none \
--sine_wave=False \
--var_q_clients=0.
```

This will run 100 communication rounds of federated training, using SGD on both
the client and server, with learning rates of 0.1 and 1.0 respectively. The
experiment uses 10 clients in each round, and performs 1 training epoch on each
client's dataset. Each client will use a batch size of 10 The `experiment_name`
flag is for the purposes of writing metrics. All clients available.

## Task and dataset summary

Below we give a summary of the datasets, tasks, and models used in this
directory.

<!-- mdformat off(This table is sensitive to automatic formatting changes) -->

Task Name | Directory        | Dataset        | Model                             | Task Summary              |
----------|------------------|----------------|-----------------------------------|---------------------------|
CIFAR-100 | cifar100         | [CIFAR-100](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)      | ResNet-18 (with GroupNorm layers) | Image classification      |
Shakespeare | shakespeare      | [Shakespeare](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)    | RNN with 2 LSTM layers            | Next-character prediction |
synthetic | synthetic     | [synthetic(1,1)](locally generated)    | Logistic Regression            | 10 class classification|

<!-- mdformat on -->

## Flags for selection policies:

```sh
--schedule=importance \
--beta=0.001  \
--experiment_name=cifar_importance_LARGE_A \
--sine_wave=False \
--var_q_clients=0.7
--schedule:[none, importance, loss ]
```

*   **none**: `--schedule=none`
*   **importance**: `--schedule=importance` --beta=0.001
*   **loss**: `--schedule=loss`

## Flags for availability models:

--sine_wave: [True, False] Wheather to use a sinusoidal modulation or not. 


--var_q_clients: [0,.0.2,0.25,0.5]

If not specified or outside the above defined parameters will generate random availabilities. 


*   **0.**:All clients available  `--var_q_clients=0.`
*   **0.2**: Small model, every client available w.p. 0.2 `--var_q_clients=0.2`
*   **0.25**: Independent model with variance 0.25 `--var_q_clients=0.25`
*   **0.5**: Independent model with variance 0.5 `--var_q_clients=0.5`

