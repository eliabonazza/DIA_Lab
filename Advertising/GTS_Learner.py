from Pricing.Learner import Learner
import numpy as np

class GTS_Learner(Learner):

    def __init__(self, n_arms):
        super(GTS_Learner, self).__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.sigma = np.ones(n_arms) * 1e3 # no prior information -> high probability to be pulled


    def pull_arm(self):
        idx = np.argmax(np.random.normal(self.means, self.sigma))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        n_samples = len(self.rewards_per_arm[pulled_arm])

        if n_samples > 1:
            self.sigma[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples

    