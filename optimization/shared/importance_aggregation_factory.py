import collections
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.common_libs import py_typecheck

import numpy as np
import attr

from typing import Callable

from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils




_ModelConstructor = Callable[[], model_lib.Model]

def weights_type_from_model_fn(
    model_fn: _ModelConstructor):
  py_typecheck.check_callable(model_fn)
  # This graph and the ones below are introduced in order to ensure that these
  # TF invocations don't leak into the global graph. In the future, it would
  # be nice if we were able to access the structure of `weights` without ever
  # actually running TF code.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_utils.weights_type_from_model(model)
  return model_weights_type.trainable




@tff.tf_computation()
def _mul(value, weight):
    return tf.nest.map_structure(lambda x: x * tf.cast(weight, x.dtype), value)

@tff.tf_computation()
def _div(value, num):
    return tf.nest.map_structure(lambda x: x/tf.cast(num, x.dtype), value)

# @attr.s(eq=False, frozen=True)
# class AggregationState(object):
#     """Structure for state on the server.

#     Fields:
#     -   `num_participants`: The number of participants, as a float.
#     """
#     num_participants = attr.ib()
 
class ImportanceSamplingFactory( tff.aggregators.WeightedAggregationFactory ):
    """`UnweightedAggregationFactory` for sum.
    The created `tff.templates.AggregationProcess` sums values placed at
    `CLIENTS`, and outputs the count placed at `SERVER`.
    The process has empty `state` and returns no `measurements`. For counting,
    implementation delegates to the `tff.federated_sum` operator.
    """

    def __init__(self,num_participants):
        self._num_participants = num_participants


    def create(self, value_type, weight_type):
        @tff.federated_computation()
        def initialize_fn():
            # state = AggregationState(self._num_participants)
            return tff.federated_value(self._num_participants, tff.SERVER)


        @tff.federated_computation(initialize_fn.type_signature.result,
                                   tff.type_at_clients(value_type),
                                      tff.type_at_clients(weight_type))
        def next_fn(state, value,weight):
            weighted_values = tff.federated_map(_mul, (value, weight))
            summed_value = tff.federated_sum(weighted_values)
            normalized_value = tff.federated_map(
                _div, (summed_value, state))
            measurements = tff.federated_value((), tff.SERVER)
            return tff.templates.MeasuredProcessOutput(
              state=state, result=normalized_value, measurements=measurements)

        return tff.templates.AggregationProcess(initialize_fn, next_fn)
