"""Use dropout for exploration for multi-armed and linear bandits."""

import numpy as np

class DropoutExploration:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.ones(self.K)  # cumulative reward
    # self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  def get_arm(self, t):
    # decision statistics
    muhat = self.reward / self.pulls
    drop_prob = self.drop_prob * np.sqrt(self.K / (t + 1)) / 2
    drop_arms = (np.random.random(self.K) >= drop_prob).astype(int)
    muhat *= drop_arms
    best_arm = np.argmax(muhat)

    # pull the arm
    arm = best_arm

    return arm

  @staticmethod
  def print():
    return "Dropout Exploration"


class LinDropout:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.n = n
    self.sigma0 = 1.0
    self.sigma = 0.5
    self.crs = 1.0 # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.Gram = 1e-3 * np.eye(self.d) / np.square(self.sigma0)
    self.B = np.zeros(self.d)

  def update(self, t, arm, r):
    x = self.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)

  def get_arm(self, t):
    self.mu = np.zeros(self.K)

    # pull each arm once at the beginning
    if t < self.K:
      self.mu[t] = np.Inf
    else:
      self.mu = np.zeros(self.K)
      drop_prob = self.drop_prob * np.sqrt(self.K / (t + 1)) / 2
      theta = np.linalg.solve(self.Gram, self.B)

      drop_dims = (np.random.random(self.d) >= drop_prob).astype(int)
      drop_dims[-1] = 1
      theta *= drop_dims

      self.mu = self.X.dot(theta)
    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Lin Dropout Exploration"


class LinDropout_gradient(LinDropout):
  def __init__(self, env, n, params):
    super().__init__(self, env, n, params)
    self.theta = np.random.rand(self.d)
    self.mask = np.ones(self.d) 
    self.lr = 0.01
    self.reg = 0.1

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    drop_prob = self.drop_prob * np.sqrt(self.K / (t + 1)) / 2
    drop_dims = (np.random.random(self.d) >= drop_prob).astype(int)
    drop_dims[-1] = 1
    #self.mask = np.ones(self.d) 
    #self.mask[drop_dims] = 0

    self.mask = drop_dims

    self.mu = self.X.dot(self.theta*self.mask)

    # if np.random.rand() < 0.05 * np.sqrt(self.n / (t + 1)) / 2:
    #   self.mu[np.random.randint(self.K)] = np.Inf
    # else:
    #   #theta = np.linalg.solve(self.Gram, self.B)
    #   self.mu = self.X.dot(self.theta)

    arm = np.argmax(self.mu)
    return arm

  def update(self, t, arm, r):
    # logistic loss
    pred = np.dot(self.theta, self.X[arm])
    grad = self.X[arm] * self.mask * (pred-r) + self.reg * self.theta

    self.theta -= self.lr * grad

  @staticmethod
  def print():
    return "Lin Dropout by Gradient Descent"






