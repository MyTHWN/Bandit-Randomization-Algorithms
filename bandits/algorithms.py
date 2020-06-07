"""Classical algorithms for multi-armed, linear, and GLM bandits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import numpy as np


class UCB1:
  def __init__(self, env, n, params):
    self.K = env.K
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.random.rand(self.K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    # UCBs
    t += 1  # time starts at one
    ciw = self.crs * np.sqrt(2 * np.log(t))
    self.ucb = self.reward / self.pulls + \
      ciw * np.sqrt(1 / self.pulls) + self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "UCB1"


class UCBV:
  def __init__(self, env, n, params):
    self.K = env.K
    self.n = n

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.reward2 = np.zeros(self.K)  # cumulative squared reward
    self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.reward2[arm] += r * r

  def get_arm(self, t):
    if t < self.K:
      # pull each arm once in the first K rounds
      self.ucb = np.zeros(self.K)
      self.ucb[t] = 1
    else:
      # UCBs
      t += 1  # time starts at one

      # from \sum_{t = 1}^n \sum_{s = 1}^t (1 / n^2) <= 1
      delta = 1.0 / np.power(self.n, 2)
      # # from \sum_{t = 1}^n \sum_{s = 1}^t (1 / t^3) <= \pi^2 / 6
      # delta = 1.0 / np.power(t, 3)

      muhat = self.reward / self.pulls
      varhat = (self.reward2 - self.pulls * np.square(muhat)) / self.pulls
      varhat = np.maximum(varhat, 0)
      self.ucb = muhat + \
        np.sqrt(2 * varhat * np.log(3 / delta) / self.pulls) + \
        3 * np.log(3 / delta) / self.pulls + \
        self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "UCB-V"


class KLUCB:
  def __init__(self, env, n, params):
    self.K = env.K

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.random.rand(self.K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

  def UCB(self, p, N, t):
    C = (np.log(t) + 3 * np.log(np.log(t) + 1e-6)) / N

    qmin = np.minimum(np.maximum(p, 1e-6), 1 - 1e-6)
    qmax = (1 - 1e-6) * np.ones(p.size)
    for i in range(16):
      q = (qmax + qmin) / 2
      ndx = (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))) < C
      qmin[ndx] = q[ndx]
      qmax[~ndx] = q[~ndx]

    return q

  def update(self, t, arm, r):
    if (r > 0) and (r < 1):
      r = (np.random.rand() < r).astype(float)
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    # UCBs
    t += 1  # time starts at one
    self.ucb = \
      self.UCB(self.reward / self.pulls, self.pulls, t) + self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "KL-UCB"


class TS:
  def __init__(self, env, n, params):
    self.K = env.K
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.alpha = np.ones(self.K)  # positive observations
    self.beta = np.ones(self.K)  # negative observations

  def update(self, t, arm, r):
    if (r > 0) and (r < 1):
      r = (np.random.rand() < r).astype(float)
    self.alpha[arm] += r
    self.beta[arm] += 1 - r

  def get_arm(self, t):
    # posterior sampling
    crs2 = np.square(self.crs)
    self.mu = np.random.beta(self.alpha / crs2, self.beta / crs2)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "TS"


class GaussTS:
  def __init__(self, env, n, params):
    self.K = env.K
    self.sigma = 0.5
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.ones(self.K)  # number of pulls
    self.reward = 0.5 * np.ones(self.K)  # cumulative reward

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    # posterior sampling
    self.mu = self.reward / self.pulls + \
      (self.sigma / (self.crs * np.sqrt(self.pulls))) * np.random.randn(self.K)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Gaussian TS"


class EpsilonGreedy:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.epsilon = self.K / np.sqrt(n)
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.ones(self.K)  # cumulative reward
    # self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

    self.grad = np.zeros(n)
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
    best_arm = np.argmax(muhat)

    # probabilities of pulling arms
    eps = self.crs * self.epsilon
    p = (1 - eps) * (np.arange(self.K) == best_arm) + eps / self.K

    # pull the arm
    arm = best_arm
    if np.random.rand() < eps:
      arm = np.random.randint(self.K)

    # derivative of the probability of the pulled arm
    self.grad[t] = self.epsilon * (1 / self.K - (arm == best_arm)) / p[arm]

    return arm

  @staticmethod
  def print():
    return "e-greedy"


class HistorySwapping:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    #self.swap_prob = swap_prob
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.ones(self.K)  # number of pulls
    self.reward = np.ones(self.K)  # cumulative reward
    # self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

    self.grad = np.zeros(n)
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)
    self.reward_hist = {i:[1] for i in range(self.K)}
    
  '''
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
  '''

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward_hist[arm].append(r)
    
    if np.random.random() < self.swap_prob:
        arm_swap = np.random.randint(self.K)
        np.random.shuffle(self.reward_hist[arm])
        np.random.shuffle(self.reward_hist[arm_swap])
        self.reward_hist[arm][0], self.reward_hist[arm_swap][0] = \
            self.reward_hist[arm_swap][0], self.reward_hist[arm][0]
        
        self.reward[arm] = np.sum(self.reward_hist[arm])
        self.reward[arm_swap] = np.sum(self.reward_hist[arm_swap])
    else: 
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
    best_arm = np.argmax(muhat)

    # probabilities of pulling arms
    # eps = self.crs * self.epsilon
    # p = (1 - eps) * (np.arange(self.K) == best_arm) + eps / self.K

    # pull the arm
    arm = best_arm
    # if np.random.rand() < eps:
      # arm = np.random.randint(self.K)

    # derivative of the probability of the pulled arm
    # self.grad[t] = self.epsilon * (1 / self.K - (arm == best_arm)) / p[arm]

    return arm

  @staticmethod
  def print():
    return "Histroy-Swapping"


class Exp3:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.crs = min(1, np.sqrt(self.K * np.log(self.K) / ((np.e - 1) * n)))

    for attr, val in params.items():
      setattr(self, attr, val)

    self.eta = self.crs / self.K
    self.reward = np.zeros(self.K)  # cumulative reward

    self.grad = np.zeros(n)
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def update(self, t, arm, r):
    self.reward[arm] += r / self.phat[arm]

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
    # probabilities of pulling arms
    scaled_reward = self.reward - self.reward.max()
    p = np.exp(self.eta * scaled_reward)
    p /= p.sum()
    self.phat = (1 - self.crs) * p + self.eta

    # pull the arm
    q = np.cumsum(self.phat)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    self.grad[t] = (1 / self.phat[arm]) * \
      ((1 - self.crs) * (p[arm] / self.K) *
      (scaled_reward[arm] - p.dot(scaled_reward)) - p[arm] + 1 / self.K)

    return arm

  @staticmethod
  def print():
    return "Exp3"


class FPL:
  def __init__(self, env, n, params):
    self.K = env.K
    self.eta = np.sqrt((np.log(self.K) + 1) / (self.K * n))

    for attr, val in params.items():
      setattr(self, attr, val)

    self.loss = np.zeros(self.K) # cumulative loss

  def update(self, t, arm, r):
    # estimate the probability of pulling the arm
    wait_time = 0
    while True:
      wait_time += 1
      ploss = self.loss + np.random.exponential(1 / self.eta, self.K)
      if np.argmin(ploss) == arm:
        break;

    self.loss[arm] += (1 - r) * wait_time

  def get_arm(self, t):
    # perturb cumulative loss
    ploss = self.loss + np.random.exponential(1 / self.eta, self.K)

    arm = np.argmin(ploss)
    return arm

  @staticmethod
  def print():
    return "FPL"


class LinBanditAlg:
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
    self.Gram = np.eye(self.d) / np.square(self.sigma0)
    self.B = np.zeros(self.d)

  def update(self, t, arm, r):
    x = self.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)


class LinUCB(LinBanditAlg):
  def __init__(self, env, n, params):
    LinBanditAlg.__init__(self, env, n, params)

    self.cew = self.crs * self.confidence_ellipsoid_width(n)

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Abassi-Yadkori (2011)
    # Improved Algorithms for Linear Stochastic Bandits
    delta = 1 / self.n
    L = np.amax(np.linalg.norm(self.X, axis=1))
    Lambda = 1 / np.square(self.sigma0)
    R = self.sigma
    S = np.sqrt(self.d)
    width = np.sqrt(Lambda) * S + \
      R * np.sqrt(self.d * np.log((1 + t * np.square(L) / Lambda) / delta))
    return width

  def get_arm(self, t):
    Gram_inv = np.linalg.inv(self.Gram)
    theta = Gram_inv.dot(self.B)

    # UCBs
    self.mu = self.X.dot(theta) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinUCB"


class LinGreedy(LinBanditAlg):
  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if np.random.rand() < 0.05 * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      theta = np.linalg.solve(self.Gram, self.B)
      self.mu = self.X.dot(theta)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Lin e-greedy"


class LinTS(LinBanditAlg):
  def get_arm(self, t):
    Gram_inv = np.linalg.inv(self.Gram)
    thetabar = Gram_inv.dot(self.B)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetabar,
      np.square(self.crs) * Gram_inv)
    self.mu = self.X.dot(thetatilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinTS"


class LogBanditAlg:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.n = n
    self.sigma0 = 1.0
    self.crs = 1.0 # confidence region scaling
    self.crs_is_width = False

    self.irls_theta = np.zeros(self.d)
    self.irls_error = 1e-3
    self.irls_num_iter = 30

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pos = np.zeros(self.K, dtype=int) # number of positive observations
    self.neg = np.zeros(self.K, dtype=int) # number of negative observations
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
    theta = np.copy(self.irls_theta)

    num_iter = 0
    while num_iter < self.irls_num_iter:
      theta_old = np.copy(theta)

      Xtheta = self.X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      pulls = self.pos + self.neg
      Gram = np.tensordot(R * pulls, self.X2, axes=([0], [0])) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * pulls * Xtheta - \
        self.pos * (self.sigmoid(Xtheta) - 1) - \
        self.neg * (self.sigmoid(Xtheta) - 0)
      theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < self.irls_error:
        break;
      num_iter += 1

    if num_iter == self.irls_num_iter:
      self.irls_theta = np.zeros(self.d)
    else:
      self.irls_theta = np.copy(theta)

    return theta, Gram


class LogUCB(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    if not self.crs_is_width:
      self.cew = self.crs * self.confidence_ellipsoid_width(n)
    else:
      self.cew = self.crs

  def confidence_ellipsoid_width(self, t):
    # Section 4.1 in Filippi (2010)
    # Parametric Bandits: The Generalized Linear Case
    delta = 1 / self.n
    c_m = np.amax(np.linalg.norm(self.X, axis=1))
    c_mu = 0.25 # minimum derivative of the mean function
    k_mu = 0.25
    kappa = np.sqrt(3 + 2 * np.log(1 + 2 * np.square(c_m / self.sigma0)))
    R_max = 1.0
    width = (2 * k_mu * kappa * R_max / c_mu) * \
      np.sqrt(2 * self.d * np.log(t) * np.log(2 * self.d * self.n / delta))
    return width

  def get_arm(self, t):
    pulls = self.pos + self.neg
    Gram = np.tensordot(pulls, self.X2, axes=([0], [0])) + \
      np.eye(self.d) / np.square(self.sigma0)
    Gram_inv = np.linalg.inv(Gram)
    theta, _ = self.solve()

    # UCBs
    self.mu = self.sigmoid(self.X.dot(theta)) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-UCB (log)"


class UCBLog(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    if not self.crs_is_width:
      self.cew = self.crs * self.confidence_ellipsoid_width(n)
    else:
      self.cew = self.crs

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Li (2017)
    # Provably Optimal Algorithms for Generalized Linear Contextual Bandits
    delta = 1 / self.n
    sigma = 0.5
    kappa = 0.25 # minimum derivative of a constrained mean function
    width = (sigma / kappa) * \
      np.sqrt((self.d / 2) * np.log(1 + 2 * self.n / self.d) + \
      np.log(1 / delta))
    return width

  def get_arm(self, t):
    pulls = self.pos + self.neg
    Gram = np.tensordot(pulls, self.X2, axes=([0], [0])) + \
      np.eye(self.d) / np.square(self.sigma0)
    Gram_inv = np.linalg.inv(Gram)
    theta, _ = self.solve()

    # UCBs
    self.mu = self.X.dot(theta) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "UCB-GLM (log)"


class LogGreedy(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    self.epsilon = 0.05

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if np.random.rand() < self.epsilon * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      theta, _ = self.solve()
      self.mu = self.sigmoid(self.X.dot(theta))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Log e-greedy"


class LogTS(LogBanditAlg):
  def get_arm(self, t):
    thetabar, Gram = self.solve()
    Gram_inv = np.linalg.inv(Gram)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetabar,
      np.square(self.crs) * Gram_inv)
    self.mu = self.X.dot(thetatilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-TSL (log)"


class LogFPL(LogBanditAlg):
  def __init__(self, env, n, params):
    self.a = 1.0

    LogBanditAlg.__init__(self, env, n, params)

  def solve(self):
    # normal noise perturbation
    pulls = self.pos + self.neg
    z = self.a * np.sqrt(pulls) * \
      np.minimum(np.maximum(np.random.randn(self.K), -6), 6)

    # iterative reweighted least squares for Bayesian logistic regression
    # Sections 4.3.3 and 4.5.1 in Bishop (2006)
    # Pattern Recognition and Machine Learning
    theta = np.copy(self.irls_theta)

    num_iter = 0
    while num_iter < self.irls_num_iter:
      theta_old = np.copy(theta)

      Xtheta = self.X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      Gram = np.tensordot(R * pulls, self.X2, axes=([0], [0])) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * pulls * Xtheta - \
        (pulls * self.sigmoid(Xtheta) - (self.pos + z))
      theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < self.irls_error:
        break;
      num_iter += 1

    if num_iter == self.irls_num_iter:
      self.irls_theta = np.zeros(self.d)
    else:
      self.irls_theta = np.copy(theta)

    return theta, Gram

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.d:
      self.mu[t] = np.Inf
    else:
      # history perturbation
      theta, _ = self.solve()
      self.mu = self.sigmoid(self.X.dot(theta))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-FPL (log)"
