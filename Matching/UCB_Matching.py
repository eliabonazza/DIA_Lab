import numpy as np
from scipy.optimize import linear_sum_assignment

from Matching.UCB import UCB


class UCB_Matching(UCB):

    def __init__(self, n_arms, n_rows, n_cols):
        super().__init__(n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        assert n_arms == n_cols * n_rows

    def pull_arm(self):
        upper_confidence = self.empirical_means + self.confidence
        upper_confidence[np.isinf(upper_confidence)] = 1e3 # large number
        row_ind, col_ind = linear_sum_assignment(
            -upper_confidence.reshape(self.n_rows, self.n_cols)
        )
        return (row_ind, col_ind)




    def update(self, pulled_arms, rewards):
        self.t += 1

        pulled_arms_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))

        for pulled_arm, reward in zip(pulled_arms_flat, rewards):
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t


        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            if n_samples > 0:
                self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5
            else:
                self.confidence[a] = np.inf


if __name__ == '__main__':
    from Pricing.Environment import Environment
    import matplotlib.pyplot as plt

    p = np.array(
        [[0.25, 1, 0.25], [0.5, 0.5, 0.25], [0.25, 0.25, 1]]
    )
    opt = linear_sum_assignment(-p)

    n_exp = 10
    T = 3000

    regret_ucb = np.zeros((n_exp, T))

    for e in range(n_exp):
        learner = UCB_Matching(p.size, *p.shape)
        print(e)
        rew_UCB = []
        opt_rew = []
        env = Environment(p.size, p)
        for t in range(T):
            pulled_arms = learner.pull_arm()
            rewards = env.round(pulled_arms)
            learner.update(pulled_arms, rewards)
            rew_UCB.append(rewards.sum())
            opt_rew.append(p[opt].sum())
        regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)

    plt.figure(0)
    plt.plot(regret_ucb.mean(axis=0))
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.show()














