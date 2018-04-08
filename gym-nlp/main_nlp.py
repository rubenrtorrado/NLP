from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari

import gym

from baselines import deepq

import gym_nlp

def main():
    logger.configure()
    set_global_seeds(0)

    env = gym.make('nlp_gym')
    env = bench.Monitor(env, logger.get_dir())
    # Remove DCNN for DRNN
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(int(1)),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=1000000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=500,
        prioritized_replay=True,
        gamma=0.99,
        print_freq=4
        #callback=callback
    )
    print("Saving model to nlp.pkl")
    act.save("nlp_model.pkl")


if __name__ == '__main__':
    main()
