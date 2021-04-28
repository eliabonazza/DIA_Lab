import numpy as np
from scipy.optimize import linear_sum_assignment

from Matching.UCB_Matching import UCB_Matching
from Matching.CUSUM import CUSUM


class CD_UCB(UCB_Matching):

    def __init__(self, n_arms, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1-self.alpha):
            upper_conf = self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e3
            row_ind, col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
            return row_ind, col_ind
        else:
            random_costs = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            return linear_sum_assignment(random_costs)

    def update(self, pulled_arms, rewards):
        self.t += 1

        pulled_arms_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))

        for pulled_arm, reward in zip(pulled_arms_flat, rewards):
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arms[pulled_arm] = []
                self.change_detection[pulled_arm].reset()


            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(
                self.valid_rewards_per_arms[pulled_arm]
            )
        total_valid_samples = sum(
            [len(x) for x in self.valid_rewards_per_arms]
        )
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            if n_samples > 0:
                self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5
            else:
                self.confidence[a] = np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from Pricing.NonStationaryEnvironment import Non_Stationary_Environment

    p0 = np.array(
        [[0.25, 1, 0.25], [0.5, 0.25, 0.25], [0.25, 0.25, 1]]
    )
    p1 = np.array(
        [[1, 0.25, 0.25], [0.5, 0.25, 0.25], [0.25, 0.25, 1]]
    )
    p2 = np.array(
        [[1, 0.25, 0.25], [0.5, 1, 0.25], [0.25, 0.25, 1]]
    )
    P = [p0,p1,p2]

    n_exp = 10
    T = 3000

    regret_cd = np.zeros((n_exp, T))
    regret_ucb = np.zeros((n_exp, T))

    detections = [[] for _ in range(n_exp)]

    M = 100
    eps = 0.1
    h = np.log(T)*2

    for e in range(n_exp):
        print(e)

        e_UCB = Non_Stationary_Environment(p0.size, P, T)
        e_CD = Non_Stationary_Environment(p0.size, P, T)

        cd_learner = CD_UCB(p0.size, *p0.shape, M, eps, h)
        ucb_learner = UCB_Matching(p0.size, *p0.shape)

        rew_CD = []
        rew_UCB = []
        opt_rew = []
        for t in range(T):
            p = P[int(t/e_UCB.phase_size)]

            opt = linear_sum_assignment(-p)
            opt_rew.append(p[opt].sum())

            pulled_arm = cd_learner.pull_arm()
            reward = e_CD.round(pulled_arm)
            cd_learner.update(pulled_arm, reward)
            rew_CD.append(reward.sum())

            pulled_arm = ucb_learner.pull_arm()
            reward = e_UCB.round(pulled_arm)
            ucb_learner.update(pulled_arm, reward)
            rew_UCB.append(reward.sum())

        regret_cd[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_CD)
        regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)

    plt.figure(0)
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.plot(np.mean(regret_cd, axis=0))
    plt.plot(np.mean(regret_ucb, axis=0))
    plt.legend(['CD-UCB','UCB'])
    plt.show()














