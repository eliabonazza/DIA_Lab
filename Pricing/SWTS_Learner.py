import numpy as np

from Pricing.TS_Learner import TS_Learner

# sliding window
class SWTS_Learner(TS_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        cum_reward = np.sum(
            self.rewards_per_arm[pulled_arm][-self.window_size:]
        )
        n_rounds_arm = len(self.rewards_per_arm[pulled_arm][-self.window_size:])

        self.beta_parameters[pulled_arm, 0] = cum_reward + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_reward + 1.0
