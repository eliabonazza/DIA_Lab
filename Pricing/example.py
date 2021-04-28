import matplotlib.pyplot as plt
import numpy as np

from Pricing.NonStationaryEnvironment import Non_Stationary_Environment
from Pricing.SWTS_Learner import SWTS_Learner

from Pricing.Environment import Environment
from Pricing.TS_Learner import TS_Learner
from Pricing.GreedyLearner import GreedyLearner


def stationary():
    n_arms = 4
    p = np.array([0.15, 0.1, 0.1, 0.35])
    opt = p[3]

    # horizon
    T = 300

    n_experiments = 100

    ts_rewards_per_experiment = []
    gr_rewards_per_experiment = []

    for e in range(n_experiments):
        env = Environment(n_arms=n_arms, probabilities=p)
        ts_learner = TS_Learner(n_arms)
        gr_learner = GreedyLearner(n_arms)
        for t in range(T):
            # ts
            pulled_arm = ts_learner.pull_arm()
            ts_reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, ts_reward)

            # gr
            pulled_arm = gr_learner.pull_arm()
            gr_reward = env.round(pulled_arm)
            gr_learner.update(pulled_arm, gr_reward)

        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        gr_rewards_per_experiment.append(gr_learner.collected_rewards)

    plt.figure(0)
    plt.xlabel('t')
    plt.ylabel('regret')
    plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
    plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
    plt.legend(['ts', 'greedy'])
    plt.show()


def non_stationary ():
    # non stationary example
    n_arms = 4
    p = np.array([[0.15, 0.1, 0.2, 0.25],
                  [0.35, 0.21, 0.2, 0.25],
                  [0.5, 0.1, 0.1, 0.15],
                  [0.8, 0.21, 0.1, 0.15]
                  ])

    T = 400
    n_experiments = 100
    ts_rewards_per_experiment = []
    swts_rewards_per_experiment = []

    window_size = int(np.sqrt(T))

    for e in range(n_experiments):
        print(e)
        ts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
        ts_learner = TS_Learner(n_arms)

        swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
        swts_learner = SWTS_Learner(n_arms, window_size)

        for t in range(T):
            # ts
            pulled_arm = ts_learner.pull_arm()
            ts_reward = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, ts_reward)

            # ts
            pulled_arm = swts_learner.pull_arm()
            swts_reward = swts_env.round(pulled_arm)
            swts_learner.update(pulled_arm, swts_reward)

        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        swts_rewards_per_experiment.append(swts_learner.collected_rewards)

    ts_instantaneous_regret = np.zeros(T)
    swts_instantaneous_regret = np.zeros(T)

    n_phases = len(p)
    phases_len = int(T / n_phases)

    opt_per_phases = p.max(axis=1)
    optimum_per_round = np.zeros(T)

    for i in range(n_phases):
        optimum_per_round[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i]
        ts_instantaneous_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(
            ts_rewards_per_experiment, axis=0)[i * phases_len: (i + 1) * phases_len]
        swts_instantaneous_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(
            swts_rewards_per_experiment, axis=0)[i * phases_len: (i + 1) * phases_len]

    plt.figure(0)
    plt.xlabel('t')
    plt.ylabel('reward')
    plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
    plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'b')
    plt.plot(optimum_per_round, '--g')
    plt.legend(['ts', 'swts'])
    plt.show()

    plt.figure(1)
    plt.xlabel('t')
    plt.ylabel('regret')
    plt.plot(np.cumsum(ts_instantaneous_regret, axis=0), 'r')
    plt.plot(np.cumsum(swts_instantaneous_regret, axis=0), 'b')
    plt.plot(optimum_per_round, 'g')
    plt.legend(['ts', 'swts'])
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stationary()


