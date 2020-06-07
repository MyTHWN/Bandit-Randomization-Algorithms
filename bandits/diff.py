"""Differentiable bandit algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import numpy as np

from google3.experimental.users.bkveton.bandits.algorithms import *


class SoftElim:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.ones(self.K)  # cumulative reward

    self.grad = np.zeros(n)
    self.metrics = np.zeros((n, 3))
    self.crs2 = self.crs ** 2
    self.crs3 = self.crs ** 3

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
    stats = - 2 * np.square((muhat.max() - muhat)) * self.pulls

    # probabilities of pulling arms
    p = np.exp(stats / self.crs2)

    # pull the arm
    q = np.cumsum(p)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    self.grad[t] = - (2 / self.crs3) * (stats[arm] -
      (stats * np.exp(stats / self.crs2)).sum() /
      np.exp(stats / self.crs2).sum())

    return arm

  @staticmethod
  def print():
    return "SoftElim"


class LinSoftElim1(LinBanditAlg):
  def __init__(self, env, n, params):
    params["sigma0"] = 0.5
    LinBanditAlg.__init__(self, env, n, params)
    self.A = LinSoftElim1.init_params()

    for attr, val in params.items():
      setattr(self, attr, val)

    self.grad = np.zeros((n, self.A.size))
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  @staticmethod
  def init_params(d=1):
    return np.ones(1)

  def update(self, t, arm, r):
    LinBanditAlg.update(self, t, arm, r)

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
    Gram_inv = np.linalg.inv(self.Gram)
    theta = Gram_inv.dot(self.B)

    # decision statistics
    # Gram_inv /= np.square(self.sigma)
    muhat = self.X.dot(theta)
    loss = np.square(muhat.max() - muhat)
    ci = np.square(self.A) * (self.X.dot(Gram_inv) * self.X).sum(axis=1)

    # arm scores
    p = np.exp(- loss / ci)
    # derivative of the log arm scores
    dlogp = 2 * loss / ci / self.A

    # pull the arm
    q = np.cumsum(p)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    self.grad[t, 0] = dlogp[arm] - (p * dlogp).sum() / p.sum()

    return arm

  @staticmethod
  def print():
    return "LinSoftElim1"


class LinSoftElim(LinBanditAlg):
  def __init__(self, env, n, params):
    params["sigma0"] = 0.5
    LinBanditAlg.__init__(self, env, n, params)
    self.A = LinSoftElim.init_params(self.d)

    for attr, val in params.items():
      setattr(self, attr, val)

    # outer products of arm features
    self.X2 = np.zeros((self.K, self.d, self.d))
    for k in range(self.K):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

    self.grad = np.zeros((n, self.A.size))
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  @staticmethod
  def init_params(d=1):
    return np.eye(d)

  def update(self, t, arm, r):
    LinBanditAlg.update(self, t, arm, r)

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
    Gram_inv = np.linalg.inv(self.Gram)
    theta = Gram_inv.dot(self.B)

    # decision statistics
    # Gram_inv /= np.square(self.sigma)
    muhat = self.X.dot(theta)
    loss = np.square(muhat.max() - muhat)
    ci = (self.X.dot(self.A.T.dot(Gram_inv).dot(self.A)) * self.X).sum(axis=1)

    # arm scores
    p = np.exp(- loss / ci)
    # derivative of the log arm scores
    # represented as [number of arms] x d x d tensor
    Gram_inv_A_X2 = np.einsum("ih,khj->kij", Gram_inv.dot(self.A), self.X2)
    dlogp = \
      2 * (loss / np.square(ci))[:, np.newaxis, np.newaxis] * Gram_inv_A_X2

    # pull the arm
    q = np.cumsum(p)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    dA = dlogp[arm, :, :] - \
      (p[:, np.newaxis, np.newaxis] * dlogp).sum(axis=0) / p.sum()
    # correction for symmetric A
    dA = dA + dA.T - np.diag(np.diag(dA))
    self.grad[t, :] = dA.flatten()

    return arm

  @staticmethod
  def print():
    return "LinSoftElim"


class CoSoftElim:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = env.Theta.shape[1]
    self.d = self.X.shape[1]
    self.num_contexts = self.X.shape[0]
    self.n = n
    self.sigma0 = 0.5
    self.sigma = 0.5
    self.A = CoSoftElim.init_params(self.d)

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    # the last dimension corresponds to arms
    self.Gram = np.zeros((self.d, self.d, self.K))
    for i in range(self.K):
      self.Gram[:, :, i] = np.eye(self.d) / np.square(self.sigma0)
    self.B = np.zeros((self.d, self.K))

    self.Gram_inv = np.zeros((self.d, self.d, self.K))
    self.Theta = np.zeros((self.d, self.K))
    for i in range(self.K):
      self.Gram_inv[:, :, i] = np.linalg.inv(self.Gram[:, :, i])
      self.Theta[:, i] = self.Gram_inv[:, :, i].dot(self.B[:, i])

    self.grad = np.zeros((n, self.A.size))
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  @staticmethod
  def init_params(d=1):
    return np.eye(d)

  def update(self, t, arm, r):
    x = self.X[self.env.ct, :]

    self.Gram[:, :, arm] += np.outer(x, x) / np.square(self.sigma)
    self.B[:, arm] += x * r / np.square(self.sigma)

    self.Gram_inv[:, :, arm] = np.linalg.inv(self.Gram[:, :, arm])
    self.Theta[:, arm] = self.Gram_inv[:, :, arm].dot(self.B[:, arm])

    best_r = self.env.rt[self.env.best_arm[self.env.ct]]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  def get_arm(self, t):
    x = self.X[self.env.ct, :]

    # decision statistics
    muhat = x.dot(self.Theta)
    loss = np.square(muhat.max() - muhat)
    Ax = self.A.dot(x)
    ci = np.einsum("jk,j->k",
      np.einsum("i,ijk->jk", Ax, self.Gram_inv), Ax)

    # arm scores
    p = np.exp(- loss / ci)
    # derivative of the log arm scores
    # represented as d x d x [number of arms] tensor
    Gram_inv_A_x2 = np.einsum("ihk,hj->ijk",
      self.Gram_inv, self.A.dot(np.outer(x, x)))
    dlogp = \
      2 * (loss / np.square(ci))[np.newaxis, np.newaxis, :] * Gram_inv_A_x2

    # pull the arm
    q = np.cumsum(p)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    dA = dlogp[:, :, arm] - \
      (p[np.newaxis, np.newaxis, :] * dlogp).sum(axis=-1) / p.sum()
    # correction for symmetric A
    dA = dA + dA.T - np.diag(np.diag(dA))
    self.grad[t, :] = dA.flatten()

    return arm

  @staticmethod
  def print():
    return "CoSoftElim"


class CoEpsilonGreedy:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = env.Theta.shape[1]
    self.d = self.X.shape[1]
    self.num_contexts = self.X.shape[0]
    self.n = n
    self.sigma0 = 0.5
    self.sigma = 0.5
    self.A = CoEpsilonGreedy.init_params(self.d)

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    # the last dimension corresponds to arms
    self.Gram = np.zeros((self.d, self.d, self.K))
    for i in range(self.K):
      self.Gram[:, :, i] = np.eye(self.d) / np.square(self.sigma0)
    self.B = np.zeros((self.d, self.K))

    self.Gram_inv = np.zeros((self.d, self.d, self.K))
    self.Theta = np.zeros((self.d, self.K))
    for i in range(self.K):
      self.Gram_inv[:, :, i] = np.linalg.inv(self.Gram[:, :, i])
      self.Theta[:, i] = self.Gram_inv[:, :, i].dot(self.B[:, i])

    self.grad = np.zeros((n, self.A.size))
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  @staticmethod
  def init_params(d=1):
    return np.ones(1)

  def update(self, t, arm, r):
    x = self.X[self.env.ct, :]

    self.Gram[:, :, arm] += np.outer(x, x) / np.square(self.sigma)
    self.B[:, arm] += x * r / np.square(self.sigma)

    self.Gram_inv[:, :, arm] = np.linalg.inv(self.Gram[:, :, arm])
    self.Theta[:, arm] = self.Gram_inv[:, :, arm].dot(self.B[:, arm])

    best_r = self.env.rt[self.env.best_arm[self.env.ct]]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  def get_arm(self, t):
    x = self.X[self.env.ct, :]

    # decision statistics
    muhat = x.dot(self.Theta)
    best_arm = np.argmax(muhat)

    # probabilities of pulling arms
    eps = 0.1
    tuned_eps = self.A * eps
    p = (1 - tuned_eps) * (np.arange(self.K) == best_arm) + tuned_eps / self.K

    # pull the arm
    q = np.cumsum(p)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    self.grad[t, 0] = eps * (1 / self.K - (arm == best_arm)) / p[arm]

    return arm

  @staticmethod
  def print():
    return "e-greedy"
