"""Low-rank environments and bandit algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import numpy as np


class LowRankBandit(object):
  """Low-rank bandit."""

  def __init__(self, U, V):
    self.U = np.copy(U)
    self.V = np.copy(V)
    self.K = self.U.shape[0]
    self.d = self.U.shape[1]
    self.L = self.V.shape[0]

    self.mu = U.dot(V.T)
    self.best_arm = self.mu.argmax(axis=1)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.X = np.random.randint(self.K)
    self.rt = (np.random.rand(self.L) < self.mu[self.X, :]).astype(float)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm[self.X]] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.X, self.best_arm[self.X]] - self.mu[self.X, arm]

  def print(self):
    return "Low-rank bandit: %d rows, %d columns, rank %d" % \
      (self.K, self.L, self.d)


class LowRankFPL(object):
  """Low-rank follow the perturbed leader."""

  def __init__(self, env, n, params):
    self.env = env
    self.n = n
    self.K = self.env.U.shape[0]
    self.d = self.env.U.shape[1]
    self.L = self.env.V.shape[0]

    self.a = 0.5  # perturbation scale (Gaussian noise)
    self.num_models = 3  # ensemble size
    self.lr = 1.0 / np.sqrt(self.n)  # learning rate
    self.batch_size = 128  # mini-batch size
    self.init_pulls = 2 * self.d  # number of initial random pulls per row

    for attr, val in params.iteritems():
      setattr(self, attr, val)

    # sufficient statistics
    self.model_size = self.n  # model size
    self.pulls = np.zeros((self.model_size, 2), dtype=int)
    self.row_pulls = np.zeros(self.K, dtype=int)
    self.y = self.a * np.minimum(np.maximum(
      np.random.randn(self.num_models, self.model_size), -6), 6)

    # low-rank factorization
    self.U = np.random.rand(self.num_models, self.K, self.d)
    self.V = np.random.rand(self.num_models, self.L, self.d)

  def update(self, t, arm, r):
    for h in range(self.num_models):
      self.y[h, t] += r
    self.pulls[t, :] = [self.env.X, arm]
    self.row_pulls[self.env.X] += 1

  def get_arm(self, t):
    self.mu = np.zeros(self.L)
    if self.row_pulls[self.env.X] < self.init_pulls:
      self.mu[np.random.randint(self.L)] = np.Inf
    else:
      sub = np.random.randint(t, size=self.batch_size)

      row_update = True
      for t in sub:
        i = self.pulls[t, 0]
        j = self.pulls[t, 1]
        yhat = (self.U[:, i, :] * self.V[:, j, :]).sum(axis=1)
        if row_update:
          self.U[:, i, :] -= self.lr * \
            (yhat - self.y[:, t])[:, np.newaxis] * self.V[:, j, :]
        else:
          self.V[:, j, :] -= self.lr * \
            (yhat - self.y[:, t])[:, np.newaxis] * self.U[:, i, :]
        row_update = not row_update

      h = np.random.randint(self.num_models)
      self.mu = self.U[h, self.env.X, :].dot(self.V[h, :, :].T)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LowRankFPL"


class LowRankFL(LowRankFPL):
  """Low-rank follow the leader."""

  def __init__(self, env, n, params):
    params["a"] = 0
    params["num_models"] = 1
    LowRankFPL.__init__(self, env, n, params)

  @staticmethod
  def print():
    return "LowRankFL"
