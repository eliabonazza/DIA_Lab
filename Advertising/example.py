import numpy as np
import matplotlib.pyplot as plt
from BiddingEnvironment import BiddingEnvironment
from GP_TS_Learner import GP_TS_Learner
from GTS_Learner import GTS_Learner



if __name__ == '__main__':
    n_arms = 20
    min_bid = 0.05
    max_bid = 1.0


    bids = np.linspace(min_bid, max_bid, n_arms)

    sigma = 10

    T = 60

    n_exp = 100 

    gts_reward_per_exp = []
    gp_ts_reward_per_exp = []

    for e in range(n_exp):
        env = BiddingEnvironment(bids, sigma)
        gts_learner = GTS_Learner(n_arms)
        gp_ts_learner = GP_TS_Learner(n_arms, arms=bids)
        for t in range(T):
            pulled_arm = gts_learner.pull_arm()
            reward = env.round(pulled_arm)
            gts_learner.update(pulled_arm, reward)

            pulled_arm = gp_ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            gp_ts_learner.update(pulled_arm, reward)

        gp_ts_reward_per_exp.append(gp_ts_learner.collected_rewards)
        gts_reward_per_exp.append(gts_learner.collected_rewards)

    opt = np.max(env.means)
    plt.figure(0)
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.plot(np.cumsum(
        np.mean(opt - gts_reward_per_exp, axis=0)
    ), 'r')
    plt.plot(np.cumsum(
        np.mean(opt - gp_ts_reward_per_exp, axis=0)
    ), 'g')

    plt.legend(['GTS','GP_TS'])
    plt.show()