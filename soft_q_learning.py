import numpy as np
import tensorflow as tf
from .utils import SimpleReplayBuffer as ReplayPool
from .kernel import adaptive_isotropic_gaussian_kernel as default_kernel


class SoftQLearning(object):

    def __init__(
            self,
            env,
            policy,
            q_function,
            render=True,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            policy_learning_rate=1e-3,
            q_func_learning_rate=1e-3,
            target_update_interval=1,
            n_action_samples_qf=16,
            n_action_samples_policy=16,
            policy_update_ratio=0.5,
            kernel_fn=default_kernel,
    ):
        self.env = env
        self.policy = policy
        self.qf = q_function
        self.render = render

        self.batch_size = batch_size
        self.n_epoch = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size

        self._n_action_samples_qf = n_action_samples_qf
        self._n_action_samples_policy = n_action_samples_policy
        self._policy_update_ratio = policy_update_ratio

        self.policy_lr = policy_learning_rate
        self.qf_lr = q_func_learning_rate

        self._target_update_interval = target_update_interval

        self.es = None
        self._reward_scale = 1

        self._kernel_fn = kernel_fn

        self._training_ops = list()
        self._create_placeholders()
        self._create_qf_update_op()
        self._create_policy_update_op()
        self._create_target_ops()

        self._sess = tf.get_default_session()
        self._sess.run(tf.global_variables_initializer())

    def _create_placeholders(self):
        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations'
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='next_observations'
        )
        self._actions_ph = tf.placeholder(tf.float32, shape=[None, self._action_dim], name='actions')
        self._next_actions_ph = tf.placeholder(tf.float32, shape=[None, self._action_dim], name='next_actions')
        self._rewards_ph = tf.placeholder(tf.float32, shape=[None], name='rewards')
        self._terminals_pl = tf.placeholder(tf.float32, shape=[None], name='terminals')

    def train(self):
        pool = ReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.env.observation_space.flat_dim,
            action_dim=self.env.action_space.flat_dim
        )

        terminal = False
        observation = self.env.reset()
        path_length = 0
        path_return = 0
        itr = 0

        for epoch in range(self.n_epoch):
            print('Starting epoch #%d' % epoch)
            for epoch_itr in range(self.epoch_length):

                if terminal:
                    observation = self.env.reset()
                    path_length = 0

                action = self.es.get_action(itr, observation, policy=self.policy)
                next_observation, reward, terminal, _ = self.env.step(action)
                path_length += 1
                path_return += 1

                if not terminal and path_length >= self.max_path_length:
                    terminal = True

                pool.add_sample(observation, action, reward, terminal)
                observation = next_observation

                if pool.size >= self.min_pool_size:
                    batch = pool.random_batch(self.batch_size)
                    self.do_training(itr, batch)

                itr += 0

    def _create_qf_update_op(self):

        with tf.variable_scope('q_target'):
            target_actions = tf.random_uniform(
                shape=(1, self._n_action_samples_qf, self._action_dim),
                min=-1,
                max=1,
            )
            q_value_targets = self.qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=target_actions
            )

        self._q_values = self.qf.output_for(self._observations_ph, self.action_ph, reuse=True)
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)

        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += self._action_dim * np.log(2)

        ys = tf.stop_gradient(self._reward_scale * self._rewards_ph + (1 - self._terminals_ph) * self._discount * next_value)
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values) ** 2)

        qf_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=bellman_residual, var_list=self.qf.get_internal_params())
        self._training_ops.append(qf_train_op)

        self._bellman_residual = bellman_residual

    def _create_policy_update_op(self):
        actions = self.policy.actions_for(
            observations=self._observations_ph,
            n_action_samples=self._n_action_samples_policy,
            reuse=True
        )

        # Split the sampled action to two sets
        # Appendix C1.1.
        n_updated_actions = int(self._n_action_samples_policy * self._policy_update_ratio)
        n_fixed_actions = self._n_action_samples_policy - n_updated_actions
        fixed_actions, updated_actions = tf.split(actions, [n_fixed_actions, n_updated_actions], axis=1)

        svgd_target_values = self.qf.output_for(self._observations_ph[:, None, :], fixed_actions, reuse=True)
        squash_correction = tf.reduce_sum(tf.log(1 - fixed_actions ** 2 + 1e-6), axis=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            updated_actions,
            self.policy.get_internal_params(),
            grad_ys=action_gradients
        )

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_internal_params(), gradients)
        ])

        optimizer = tf.train.AdamOptimizer(self._policy_lr)
        svgd_training_op = optimizer.minimize(
            loss=-surrogate_loss,
            var_list=self.policy.get_internal_params())
        self._training_ops.append(svgd_training_op)

    def _create_target_ops(self):

        source_params = self.qf.get_params_internal()
        target_params = self.qf.get_params_internal(scope='target')

        self._target_ops = [
            tf.assign(tgt, src)
            for tgt, src in zip(target_params, source_params)
        ]

    def _do_training(self, iter, batch):

        feeds = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_pl: batch['terminals'],
        }

        self._sess.run(self._training_ops, feeds)

        if iter % self._target_update_interval:
            self._sess.run(self._target_ops)













