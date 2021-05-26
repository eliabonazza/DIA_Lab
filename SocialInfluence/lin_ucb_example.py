import numpy as np
import matplotlib.pyplot as plt

from LinearMabEnvironment import LinearMabEnvironment
from LinearUCBLearner import LinearUCBLearner



if __name__ == '__main__':
    n_arms = 10
    T = 1000
    n_experiments = 100
    lin_ucb_reward_per_experiment = []

    env = LinearMabEnvironment(n_arms=n_arms, dim=10)

    for e in range(n_experiments):
        lin_ucb_learner = LinearUCBLearner(arms_features=env.arms_features)

        for t in range(T):
            pulled_arm = lin_ucb_learner.pull_arm()
            reward = env.round(pulled_arm)
            lin_ucb_learner.update(pulled_arm, reward)

        lin_ucb_reward_per_experiment.append(lin_ucb_learner.collected_rewards)

    opt = env.opt()

    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(
        np.mean(
            opt - lin_ucb_reward_per_experiment, axis=0)
        ), 'r'
    )
    plt.legend("LinearUCBLearner")
    plt.show()
