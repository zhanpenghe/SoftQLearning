import tensorflow as tf
from nn import feedforward_net


class StochasticNNPolicy(object):
    
    def __init__(
            self,
            action_dim,
            observation_dim,
            hidden_layer_sizes,
            squash=True,
            name='policy'
    ):
        super(StochasticNNPolicy, self).__init__()

        self._action_dim = action_dim
        self._observation_dim = observation_dim
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._name = name

        self._scope_name = (tf.get_variable_scope().name if not self._name else self._name)

        # Create placeholders
        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation'
        )

        # Create get_action_op
        self._actions = self.actions_for(self._observation_ph)

    def actions_for(self, observations, n_action_samples=1, reuse=False):

        n_observations = tf.shape(observations)[0]

        if n_action_samples > 1:
            # Multiple samples, expand dim of observations
            observations = observations[:, None, :]
            latent_shape = (n_observations, n_action_samples, self._action_dim)
        else:
            latent_shape = (n_observations, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
            )

        return tf.nn.tanh(raw_actions) if self._squash else raw_actions

    def get_action(self, observation):
        return self.get_actions(observation[None])[0]

    def get_actions(self, observations):
        feeds = {self._observation_ph: observations}
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions

    def get_internal_params(self):

        scope = self._scope_name
        # Append a / if not empty
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
