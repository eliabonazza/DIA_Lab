import numpy as np
from Pricing.Learner import Learner

class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    # select the arm to pull
    def pull_arm(self):
        idx = np.argmax(
            np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        )
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1.0 - reward