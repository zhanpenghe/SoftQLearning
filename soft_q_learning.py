from .utils import SimpleReplayBuffer as ReplayPool


class SoftQLearning(object):

    def __init__(
            self,
            env,
            policy,
            render=True,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            policy_learning_rate=1e-3,
            q_func_learning_rate=1e-3,
            q_target_update_interval=1,
    ):
        self.env = env
        self.policy = policy
        self.render = render

        self.batch_size = batch_size
        self.n_epoch = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size

        self.policy_lr = policy_learning_rate
        self.qf_lr = q_func_learning_rate

        self.q_target_update_interval = q_target_update_interval

        self.es = None

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

    def _do_training(self, itr, batch):
        pass












