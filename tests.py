import gym
import tensorflow as tf
from policies import StochasticNNPolicy
from value_functions import NNQFunction
from soft_q_learning import SoftQLearning


def test_training(env_name):
    env = gym.make('Swimmer-v2')

    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]

    q_function = NNQFunction(
        action_dim=action_dim,
        observation_dim=observation_dim,
        hidden_layer_sizes=[100, 100],
        name='q_function'
    )

    policy = StochasticNNPolicy(
        action_dim=action_dim,
        observation_dim=observation_dim,
        hidden_layer_sizes=[100, 100],
        squash=True,
        name='policy'
    )

    algo = SoftQLearning(
        env=env,
        action_dim=action_dim,
        observation_dim=observation_dim,
        policy=policy,
        q_function=q_function,
    )
    algo.train()


if __name__ == '__main__':

    test_training('')
