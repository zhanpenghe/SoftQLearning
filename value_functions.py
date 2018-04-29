import tensorflow as tf
from nn import feedforward_net


class NNQFunction(object):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=(100, 100),
                 name='q_function'):
        super(NNQFunction, self).__init__()

        self._action_dim = env_spec.action_space.flat_dim
        self._observation_dim = env_spec.observation_space.flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [1]
        self._name = name

        self._scope_name = (tf.get_variable_scope().name if not self._name else self._name)

        self._observations_ph = tf.placeholder(tf.float32, shape=[None, self._observation_dim], name='observations')
        self._actions_ph = tf.placeholder(tf.float32, shape=[None, self._action_dim], name='actions')

        self._inputs = (self._observations_ph, self._actions_ph)
        self._output = self._output_for(self._inputs)

    def _output_for(self, inputs, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            out = feedforward_net(
                inputs=inputs,
                layer_sizes=self._layer_sizes,
            )
        return out[..., 0]

    def _eval(self, inputs):
        feeds = {pl: val for pl, val in zip(self._inputs, inputs)}

        return tf.get_default_session().run(self._output, feeds)

    def get_internal_params(self):

        scope = self._scope_name
        # Append a / if not empty
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def eval(self, observations, actions):
        return self._eval((observations, actions))
