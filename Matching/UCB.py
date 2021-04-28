import numpy as np
from Pricing.Learner import  Learner

class UCB(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(
            np.where(
                upper_conf == upper_conf.max()
            )[0]
        )

    def update(self, pulled_arm, reward):
        self.t += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t -1) + reward) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            if n_samples > 0:
                self.confidence[a] = (a * np.log(self.t)/n_samples) ** 0.5
        self.update_observations(pulled_arm, reward)




