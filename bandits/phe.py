"""Perturbed-history exploration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import time
import numpy as np


class BerGiro:
  def __init__(self, env, n, params):
    self.K = env.K
    self.a = 1

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pos = np.zeros(self.K, dtype=int) # positive observations
    self.neg = np.zeros(self.K, dtype=int) # negative observations
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

    self.mu = np.ones(self.K) + self.tiebreak # bootstrap means

  def update(self, t, arm, r):
    self.pos[arm] += r
    self.neg[arm] += 1 - r

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.K:
      # each arm is pulled once in the first K rounds
      self.mu[t] = np.Inf
    else:
      # bootstrapping
      pulls = self.pos + self.neg
      pseudo_pulls = self.a * pulls
      floor_pulls = np.floor(pseudo_pulls).astype(int)
      rounded_pulls = floor_pulls + \
        (np.random.rand(self.K) < pseudo_pulls - floor_pulls)
      self.count = np.random.binomial(pulls + 2 * rounded_pulls, \
        (self.pos + rounded_pulls) / (pulls + 2 * rounded_pulls))
      self.mu = self.count / (pulls + 2 * rounded_pulls) + self.tiebreak

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Bernoulli Giro"

class Giro:
  def __init__(self, env, n, params):
    self.K = env.K
    self.a = 1

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K, dtype=int) # number of pulls
    self.reward = np.zeros((n, self.K)) # rewards
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

  def update(self, t, arm, r):
    self.reward[self.pulls[arm], arm] = r
    self.pulls[arm] += 1

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.K:
      # each arm is pulled once in the first K rounds
      self.mu[t] = np.Inf
    else:
      # bootstrapping
      for k in range(self.K):
        pseudo_pulls = self.a * self.pulls[k]
        floor_pulls = np.floor(pseudo_pulls).astype(int)
        rounded_pulls = floor_pulls + \
          (np.random.rand() < pseudo_pulls - floor_pulls)
        H = np.concatenate((self.reward[: self.pulls[k], k], \
          np.zeros(rounded_pulls), np.ones(rounded_pulls)))
        sub = np.random.randint(0, H.size, H.size)
        self.mu[k] = H[sub].mean() + self.tiebreak[k]

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Giro"

class PHE:
  def __init__(self, env, n, params):
    self.K = env.K
    self.a = 2
    self.pseudo_reward = "bernoulli"

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K, dtype=int) # number of pulls
    self.reward = np.zeros(self.K) # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t
      return arm

    else:
      # history perturbation
      pseudo_pulls = np.ceil(self.a * self.pulls).astype(int)
      if self.pseudo_reward == "bernoulli":
        pseudo_reward = np.random.binomial(pseudo_pulls, 0.5)
      else:
        pseudo_reward = np.random.normal(0.5 * pseudo_pulls, \
                                         0.5 * np.sqrt(pseudo_pulls))
      self.mu = (self.reward + pseudo_reward) / \
        (self.pulls + pseudo_pulls) + self.tiebreak

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "PHE"

class LinPHE:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.a = 2

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pulls = np.zeros(self.K, dtype=int) # number of pulls
    self.reward = np.zeros(self.K) # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
    self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
    for k in range(self.K):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.K:
      arm = t
      return arm
      
    else:
      # history perturbation
      pseudo_pulls = np.ceil(self.a * self.pulls).astype(int)
      if self.env.noise == "normal":
        pseudo_reward = np.random.normal(self.env.sigma * pseudo_pulls, \
                                  self.env.sigma * np.sqrt(pseudo_pulls))
      else:
        pseudo_reward = np.random.binomial(pseudo_pulls, 0.5)

      Gram = np.tensordot(self.pulls + pseudo_pulls, self.X2, \
        axes=([0], [0]))
      B = self.X.T.dot(self.reward + pseudo_reward)

      reg = 1e-3 * np.eye(self.d)
      theta = np.linalg.solve(Gram + reg, B)
      self.mu = self.X.dot(theta) + self.tiebreak

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinPHE"


class CoLinPHE:
  def __init__(self, env, n, params):
    self.env = env
    self.K = self.env.X.shape[0]
    self.d = self.env.X.shape[1]
    self.a = 2 # number of pseudo-rewards per observed reward
    self.batch_size = n # maximum batch size in regression problems
    self.ninit = self.d # number of initial random pulls

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.X = np.zeros((n, self.d)) # observed features
    self.y = np.zeros(n) # observed rewards
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

  def update(self, t, arm, r):
    self.X[t, :] = self.env.X[arm, :]
    self.y[t] = r

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.ninit:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      # history perturbation
      if t < self.batch_size:
        sub = np.arange(t)
      else:
        sub = np.random.randint(t, size=self.batch_size)
      Xp = np.tile(self.X[sub, :], (self.a + 1, 1))
      yp = np.concatenate((self.y[sub], \
        np.random.randint(0, 2, self.a * sub.size)))

      reg = 1e-3 * np.eye(self.d)
      theta = np.linalg.solve(Xp.T.dot(Xp) + reg, Xp.T.dot(yp))
      self.mu = self.env.X.dot(theta) + self.tiebreak

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "CoLinPHE"

class LogPHE:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.sigma0 = 1
    self.a = 2

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pos = np.zeros(self.K, dtype=int) # number of positive observations
    self.neg = np.zeros(self.K, dtype=int) # number of negative observations
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
    self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
    for k in range(self.K):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

  def update(self, t, arm, r):
    self.pos[arm] += r
    self.neg[arm] += 1 - r

  def sigmoid(self, x):
    y = 1 / (1 + np.exp(- x))
    return y

  def solve(self):
    # iterative reweighted least squares for Bayesian logistic regression
    # Sections 4.3.3 and 4.5.1 in Bishop (2006)
    # Pattern Recognition and Machine Learning
    theta = np.zeros(self.d)
    num_iter = 0
    while num_iter < 100:
      theta_old = np.copy(theta)

      Xtheta = self.X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      pulls = self.posp + self.negp
      Gram = np.tensordot(R * pulls, self.X2, axes=([0], [0])) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * pulls * Xtheta - \
        self.posp * (self.sigmoid(Xtheta) - 1) - \
        self.negp * (self.sigmoid(Xtheta) - 0)
      theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < 1e-3:
        break;
      num_iter += 1

    return theta, Gram

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.d:
      self.mu[t] = np.Inf
    else:
      # history perturbation
      pulls = self.pos + self.neg
      pseudo_pulls = np.ceil(self.a * pulls).astype(int)
      pseudo_reward = np.random.binomial(pseudo_pulls, 0.5)
      self.posp = self.pos + pseudo_reward
      self.negp = self.neg + pseudo_pulls - pseudo_reward

      theta, _ = self.solve()
      self.mu = self.sigmoid(self.X.dot(theta)) + self.tiebreak

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LogPHE"


class CoLogPHE:
  def __init__(self, env, n, params):
    self.env = env
    self.K = self.env.X.shape[0]
    self.d = self.env.X.shape[1]
    self.sigma0 = 1
    self.a = 2 # number of pseudo-rewards per observed reward
    self.batch_size = n # maximum batch size in regression problems
    self.ninit = self.d # number of initial random pulls

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.X = np.zeros((n, self.d)) # observed features
    self.y = np.zeros(n) # observed rewards
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

  def update(self, t, arm, r):
    self.X[t, :] = self.env.X[arm, :]
    self.y[t] = r

  def sigmoid(self, x):
    y = 1 / (1 + np.exp(- x))
    return y

  def solve(self, X, y):
    # iterative reweighted least squares for Bayesian logistic regression
    # Sections 4.3.3 and 4.5.1 in Bishop (2006)
    # Pattern Recognition and Machine Learning
    theta = np.zeros(self.d)
    num_iter = 0
    while num_iter < 100:
      theta_old = np.copy(theta)

      Xtheta = X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      Gram = (X * R[:, np.newaxis]).T.dot(X) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * Xtheta - (self.sigmoid(Xtheta) - y)
      theta = np.linalg.solve(Gram, X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < 1e-3:
        break;
      num_iter += 1

    return theta, Gram

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.ninit:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      # history perturbation
      if t < self.batch_size:
        sub = np.arange(t)
      else:
        sub = np.random.randint(t, size=self.batch_size)
      Xp = np.tile(self.X[sub, :], (self.a + 1, 1))
      yp = np.concatenate((self.y[sub], \
        np.random.randint(0, 2, self.a * sub.size)))

      theta, _ = self.solve(Xp, yp)
      self.mu = self.sigmoid(self.env.X.dot(theta)) + self.tiebreak

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "CoLogPHE"
