from Pricing.Environment import Environment
import numpy as np

class Non_Stationary_Environment(Environment):
  def __init__(self, n_arms, probabilities, horizon):
    super().__init__(n_arms, probabilities)
    self.horizon = horizon
    self.t = 0
    self.phase_size = self.horizon / len(self.probabilities)

  def round(self, pulled_arm):
    n_phases = len(self.probabilities)
    phase_size = self.horizon / n_phases
    current_phase = int(self.t / phase_size)

    p = self.probabilities[current_phase][pulled_arm]
    self.t += 1
    return np.random.binomial(1, p)
