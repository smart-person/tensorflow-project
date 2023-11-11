# Copyright 2020, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared library for setting up federated training experiments."""

import collections
import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


#  Settings for a multiplicative linear congruential generator (aka Lehmer
#  generator) suggested in 'Random Number Generators: Good
#  Ones are Hard to Find' by Park and Miller.
MLCG_MODULUS = 2**(31) - 1
MLCG_MULTIPLIER = 16807


# TODO(b/143440780): Create more comprehensive tuple conversion by adding an
# explicit namedtuple checking utility.
def convert_to_tuple_dataset(dataset):
  """Converts a dataset to one where elements have a tuple structure.

  Args:
    dataset: A `tf.data.Dataset` where elements either have a mapping
      structure of format {"x": <features>, "y": <labels>}, or a tuple-like
        structure of format (<features>, <labels>).

  Returns:
    A `tf.data.Dataset` object where elements are tuples of the form
    (<features>, <labels>).

  """
  example_structure = dataset.element_spec
  if isinstance(example_structure, collections.Mapping):
    # We assume the mapping has `x` and `y` keys.
    convert_map_to_tuple = lambda example: (example['x'], example['y'])
    try:
      return dataset.map(convert_map_to_tuple)
    except:
      raise ValueError('For datasets with a mapping structure, elements must '
                       'have format {"x": <features>, "y": <labels>}.')
  elif isinstance(example_structure, tuple):

    if hasattr(example_structure, '_fields') and isinstance(
        example_structure._fields, collections.Sequence) and all(
            isinstance(f, str) for f in example_structure._fields):
      # Dataset has namedtuple structure
      convert_tuplelike_to_tuple = lambda x: (x[0], x[1])
    else:
      # Dataset does not have namedtuple structure
      convert_tuplelike_to_tuple = lambda x, y: (x, y)

    try:
      return dataset.map(convert_tuplelike_to_tuple)
    except:
      raise ValueError('For datasets with tuple-like structure, elements must '
                       'have format (<features>, <labels>).')
  else:
    raise ValueError(
        'Expected evaluation dataset to have elements with a mapping or '
        'tuple-like structure, found {} instead.'.format(example_structure))


def build_evaluate_fn(
    eval_dataset: tf.data.Dataset, model_builder: Callable[[], tf.keras.Model],
    loss_builder: Callable[[], tf.keras.losses.Loss],
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]]
) -> Callable[[tff.learning.ModelWeights], Dict[str, Any]]:
  """Builds an evaluation function for a given model and test dataset.

  The evaluation function takes as input a fed_avg_schedule.ServerState, and
  computes metrics on a keras model with the same weights.

  Args:
    eval_dataset: A `tf.data.Dataset` object. Dataset elements should either
      have a mapping structure of format {"x": <features>, "y": <labels>}, or a
        tuple structure of format (<features>, <labels>).
    model_builder: A no-arg function that returns a `tf.keras.Model` object.
    loss_builder: A no-arg function returning a `tf.keras.losses.Loss` object.
    metrics_builder: A no-arg function that returns a list of
      `tf.keras.metrics.Metric` objects.

  Returns:
    A function that take as input a `tff.learning.ModelWeights` and returns
    a dict of (name, value) pairs for each associated evaluation metric.
  """

  def compiled_eval_keras_model():
    model = model_builder()
    model.compile(
        loss=loss_builder(),
        optimizer=tf.keras.optimizers.SGD(),  # Dummy optimizer for evaluation
        metrics=metrics_builder())
    return model

  eval_tuple_dataset = convert_to_tuple_dataset(eval_dataset)

  def evaluate_fn(reference_model: tff.learning.ModelWeights) -> Dict[str, Any]:
    """Evaluation function to be used during training."""

    if not isinstance(reference_model, tff.learning.ModelWeights):
      raise TypeError('The reference model used for evaluation must be a'
                      '`tff.learning.ModelWeights` instance.')

    keras_model = compiled_eval_keras_model()
    reference_model.assign_weights_to(keras_model)
    logging.info('Evaluating the current model')
    eval_metrics = keras_model.evaluate(eval_tuple_dataset, verbose=0)
    return dict(zip(keras_model.metrics_names, eval_metrics))

  return evaluate_fn

def build_unweighted_test_fn(
    federated_eval_dataset: tff.simulation.ClientData, 
    model_builder: Callable[[], tf.keras.Model],
    loss_builder: Callable[[], tf.keras.losses.Loss],
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]]
) -> Callable[[tff.learning.ModelWeights], Dict[str, Any]]:
  """Builds an evaluation function for a given model and test dataset.

  The evaluation function takes as input a fed_avg_schedule.ServerState, and
  computes metrics on a keras model with the same weights.

  Args:
    eval_dataset: A `tf.data.Dataset` object. Dataset elements should either
      have a mapping structure of format {"x": <features>, "y": <labels>}, or a
        tuple structure of format (<features>, <labels>).
    model_builder: A no-arg function that returns a `tf.keras.Model` object.
    loss_builder: A no-arg function returning a `tf.keras.losses.Loss` object.
    metrics_builder: A no-arg function that returns a list of
      `tf.keras.metrics.Metric` objects.

  Returns:
    A function that take as input a `tff.learning.ModelWeights` and returns
    a dict of (name, value) pairs for each associated evaluation metric.
  """

  def compiled_eval_keras_model():
    model = model_builder()
    model.compile(
        loss=loss_builder(),
        optimizer=tf.keras.optimizers.SGD(),  # Dummy optimizer for evaluation
        metrics=metrics_builder())
    return model

  client_ids = federated_eval_dataset.client_ids

  def evaluate_fn(reference_model: tff.learning.ModelWeights) -> Dict[str, Any]:
    """Evaluation function to be used during training."""

    if not isinstance(reference_model, tff.learning.ModelWeights):
      raise TypeError('The reference model used for evaluation must be a'
                      '`tff.learning.ModelWeights` instance.')

    keras_model = compiled_eval_keras_model()
    reference_model.assign_weights_to(keras_model)
    logging.info('Evaluating the current model')
    for i,client_id in enumerate(client_ids):
      if i==0:
        results={}
        client_data = federated_eval_dataset.create_tf_dataset_for_client(client_id)
        eval_tuple_dataset = convert_to_tuple_dataset(client_data)
        eval_metrics = keras_model.evaluate(eval_tuple_dataset, verbose=0)
        for i,name in enumerate(keras_model.metrics_names):
          results[name]=[]
          results[name].append(eval_metrics[i])
      else:
        client_data = federated_eval_dataset.create_tf_dataset_for_client(client_id)
        eval_tuple_dataset = convert_to_tuple_dataset(client_data)
        eval_metrics = keras_model.evaluate(eval_tuple_dataset, verbose=0)
        for i,name in enumerate(keras_model.metrics_names):
          results[name].append(eval_metrics[i])
    statistics_dict = {}
    for name in keras_model.metrics_names:
      statistics_dict[f'avg_{name}'] = np.mean(results[name])
      statistics_dict[f'min_{name}'] = np.min(results[name])
      statistics_dict[f'max_{name}'] = np.max(results[name])
      statistics_dict[f'std_{name}']= np.std(results[name])
    return statistics_dict
  return evaluate_fn


def build_sample_fn(
    a: Union[Sequence[Any], int],
    p_vector:Union[Sequence[Any], int],
    size: int,
    f_distribution,
    q_client,
    num_clients,
    replace: bool = False,
    use_p: bool= False,
    random_seed: Optional[int] = None) -> Callable[[int], np.ndarray]:
  """Builds the function for sampling from the input iterator at each round.

  Args:
    a: A 1-D array-like sequence or int that satisfies np.random.choice.
    size: The number of samples to return each round.
    replace: A boolean indicating whether the sampling is done with replacement
      (True) or without replacement (False).
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process.

  Returns:
    A function which returns a list of elements from the input iterator at a
    given round round_num.
  """

  if isinstance(random_seed, int):
    mlcg_start = np.random.RandomState(random_seed).randint(1, MLCG_MODULUS - 1)

    def get_pseudo_random_int(round_num):
      return pow(MLCG_MULTIPLIER, round_num,
                 MLCG_MODULUS) * mlcg_start % MLCG_MODULUS

  def sample(round_num, random_seed):
    time = round_num%24
    time_availability = f_distribution[time]
    probs = q_client*time_availability
    available_clients = []

    #while len(available_clients)<size:
    availability = tf.random.stateless_binomial(shape=(num_clients,), 
                                                  seed=[1,round_num],
                                                  counts=tf.ones(num_clients), 
                                                  probs = probs, 
                                                  output_dtype=tf.float32)
    available_clients = [id_ for i,id_ in enumerate(a) if availability[i]]
      #logging.info(f'AVAIL: {len(available_clients)}')
    if use_p:
      probs_data = p_vector[[i for i,id_ in enumerate(a) if availability[i]]]
      probs_data = probs_data/np.sum(probs_data)
      #logging.info(f'probs_data_shape: {probs_data.shape}')
    else:
      probs_data = np.repeat(1/len(available_clients), len(available_clients))

    if isinstance(random_seed, int):
      random_state = np.random.RandomState(get_pseudo_random_int(round_num))
    else:
      random_state = np.random.RandomState()
    return random_state.choice(available_clients, size=size, replace=replace, p=probs_data),availability

  return functools.partial(sample, random_seed=random_seed)

def build_client_datasets_fn(
    train_dataset: tff.simulation.ClientData,
    train_clients_per_round: int,
    random_seed: Optional[int] = None,
    min_clients: Optional[int] = 50,
    sine_wave:Optional[bool] = True,
    var_q_clients: Optional[float] = 0.25,
    f_mult: Optional[float] = 0.4,
    f_intercept: Optional[float] = 0.5,
    use_p: Optional[bool] = False,
    q_client: Optional[List[float]]=None) -> Callable[[int], List[tf.data.Dataset]]:
  """Builds the function for generating client datasets at each round.

  The function samples a number of clients (without replacement within a given
  round, but with replacement across rounds) and returns their datasets.

  Args:
    train_dataset: A `tff.simulation.ClientData` object.
    train_clients_per_round: The number of client participants in each round.
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process. Note that this will alter the global numpy random seed.

  Returns:
    A function which returns a list of `tf.data.Dataset` objects at a
    given round round_num.
  """
  NUM_CLIENTS = len(train_dataset.client_ids)
  times = np.linspace(start=0, stop=2*np.pi, num=24)
  logging.info(f'Using sine wave:  {sine_wave}')

  p_vector = [ ]
  for client_id in train_dataset.client_ids:
    dataset = train_dataset.create_tf_dataset_for_client(client_id)
    p_vector.append(len(list(dataset)))
  p_vector = np.array(p_vector)/sum(p_vector)


  if sine_wave:
    f_distribution = np.sin(times)*f_mult+f_intercept # range between 0 - 1
  else:
    f_distribution = np.ones_like(times)
  
  if q_client is None:
    logging.info(' Created q inverse to dataset size ')
    q_client = 1/p_vector
    q_client=q_client/max(q_client)
    # raise ValueError(' q is None! ')
    # created_q = False
    # trials=0
    # while  not created_q and trials<5:
    #   logging.info(' creating q')
    #   q_client = np.random.lognormal(0., var_q_clients, (NUM_CLIENTS))
    #   q_client = q_client/max(q_client)
    #   logging.info(f'trial {trials}   -  lognormal Variance: {var_q_clients} - participating clients in min round: {sum(q_client)*f_distribution[17]}')
    #   trials+=1
    #   if sum(q_client)*f_distribution[17]>min_clients:
    #     created_q=True
    # if trials>=5:
    #   raise ValueError('Could not create q! decrease var q!')


  sample_clients_fn = build_sample_fn(
      train_dataset.client_ids,
      size=train_clients_per_round,
      replace=True,
      f_distribution=f_distribution,
      q_client=q_client,
      num_clients = NUM_CLIENTS,
      random_seed=random_seed, 
      use_p=use_p,
      p_vector=p_vector)

  def client_datasets(round_num):
    sampled_clients, availability = sample_clients_fn(round_num)
    # logging.info(f'sampled {len(sampled_clients)} clients out of {sum(availability)} available')
    datasets = [
        train_dataset.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]
    #logging.info(f'BUILD {len(datasets)} DATASETS! ')
    return datasets, availability, sampled_clients

  return client_datasets

@tf.function
def negative_variance_fn(rk, pk):
    pk_squared = tf.math.multiply(pk,pk)
    ratio = tf.math.divide_no_nan(pk_squared,rk)
    return -tf.reduce_sum(ratio)

def update_vectors(r_vector,p_vector,availability, train_clients_per_round, beta):
    # print(f'r: {r_vector.shape}, p: {p_vector.shape}')
    # print(r_vector)
    # print(p_vector)
    with tf.GradientTape() as tape:
        var = negative_variance_fn(rk=r_vector,pk=p_vector)

    dv_dr = tape.gradient(var,r_vector)
    rule = tf.multiply(dv_dr, availability)
    _, idx_ids = tf.math.top_k(rule, k=train_clients_per_round)

    r_vector.assign(tf.tensor_scatter_nd_add((1-beta)*r_vector, 
                                  tf.reshape(idx_ids,[train_clients_per_round,1] ), 
                                  beta*tf.ones(train_clients_per_round)))
    return r_vector, idx_ids
    

def build_availability_client_datasets_fn(
    train_dataset: tff.simulation.ClientData,
    train_clients_per_round: int,
    beta,
    random_seed: Optional[int] = None,
    min_clients: Optional[int] = 50,
    var_q_clients: Optional[int] = 0.25,
    sine_wave:Optional[bool] = True,
    f_mult: Optional[float] = 0.4,
    f_intercept: Optional[float] = 0.5,
    q_client: Optional[List[float]]=None,
    initialize_p=True,
) -> Callable[[int], List[tf.data.Dataset]]:
  """Builds the function for generating client datasets at each round.

  The function samples a number of clients (without replacement within a given
  round, but with replacement across rounds) and returns their datasets.

  Args:
    train_dataset: A `tff.simulation.ClientData` object.
    train_clients_per_round: The number of client participants in each round.
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process. Note that this will alter the global numpy random seed.

  Returns:
    A function which returns a list of `tf.data.Dataset` objects at a
    given round round_num.
  """

  #Define availability parameters
  NUM_CLIENTS = len(train_dataset.client_ids)
  times = np.linspace(start=0, stop=2*np.pi, num=24)
  if sine_wave:
    f_distribution = np.sin(times)*f_mult+f_intercept # range between 0 - 1
  else:
    f_distribution = np.ones_like(times)
  
  p_vector = [ ]
  for client_id in train_dataset.client_ids:
    dataset = train_dataset.create_tf_dataset_for_client(client_id)
    p_vector.append(len(list(dataset)))
  p_vector = np.array(p_vector)/sum(p_vector)

  p_vector = tf.constant(p_vector, dtype = tf.float32)  
  if q_client is None:
    logging.info(' Created q inverse to dataset size ')
    q_client = 1/p_vector
    q_client=q_client/max(q_client)
    # raise ValueError(' q is None! ')
    # created_q = False
    # trials=0
    # while  not created_q and trials<5:
    #   logging.info(' creating q')
    #   q_client = np.random.lognormal(0., var_q_clients, (NUM_CLIENTS))
    #   q_client = q_client/max(q_client)
    #   logging.info(f'trial {trials}   -  participating clients: {sum(q_client)*f_distribution[17]}')
    #   trials+=1
    #   if sum(q_client)*f_distribution[17]>min_clients:
    #     created_q=True
    # if trials>=5:
    #   raise ValueError('Could not create q! decrease var q!')

  def client_datasets(round_num,r_vector=None):
    if r_vector is None and initialize_p:
      r_vector = tf.Variable(p_vector, dtype = tf.float32)
    elif r_vector is None and not initialize_p:
      r_vector = tf.Variable(tf.ones_like(p_vector)*1/p_vector.shape[0], dtype = tf.float32)
    time = round_num%24
    time_availability = f_distribution[time]
    probs = q_client*time_availability

    availability = tf.random.stateless_binomial(shape=(NUM_CLIENTS,), 
                                                seed=[1,round_num],
                                                counts=tf.ones(NUM_CLIENTS), 
                                                probs = probs, 
                                                output_dtype=tf.float32)
    r_vector, idx_ids = update_vectors(r_vector,p_vector,availability, train_clients_per_round, beta)
    datasets = [
        train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[client])
        for client in idx_ids
    ]
    selected_p_vector = tf.gather(p_vector, idx_ids)
    selected_r_vector = tf.gather(r_vector, idx_ids)
    weights = tf.math.divide_no_nan(selected_p_vector, selected_r_vector)
    return datasets, weights, r_vector, idx_ids, availability

  return client_datasets
