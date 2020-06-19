"""Bandit simulator and environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import time
import numpy as np
import multiprocessing as mp


class BerBandit(object):
  """Bernoulli bandit."""

  def __init__(self, mu):
    self.mu = np.copy(mu)
    self.K = self.mu.size

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = (np.random.rand(self.K) < self.mu).astype(float)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Bernoulli bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class BetaBandit(object):
  """Beta bandit."""

  def __init__(self, mu, a_plus_b=4):
    self.mu = np.copy(mu)
    self.K = self.mu.size
    self.a_plus_b = a_plus_b

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = \
      np.random.beta(self.a_plus_b * self.mu, self.a_plus_b * (1 - self.mu))

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Beta bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class GaussBandit(object):
  """Gaussian bandit."""

  def __init__(self, mu, sigma=0.5):
    self.mu = np.copy(mu)
    self.K = self.mu.size
    self.sigma = sigma

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = self.mu + self.sigma * np.random.randn(self.K)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Gaussian bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


#need to set the type of noise in Linear bandits
class LinBandit(object):
  """Linear bandit."""

  def __init__(self, X, theta, noise="normal", sigma=0.2):
    self.X = np.copy(X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.theta = np.copy(theta)
    self.noise = noise
    if self.noise == "normal":
      self.sigma = sigma

    self.mu = self.X.dot(self.theta)
    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    if self.noise == "normal":
      self.rt = self.mu + self.sigma * np.random.randn(self.K)
    elif self.noise == "bernoulli":
      self.rt = (np.random.rand(self.K) < self.mu).astype(float)
    elif self.noise == "beta":
      self.rt = np.random.beta(4 * self.mu, 4 * (1 - self.mu))

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    if self.noise == "normal":
      return "Linear bandit: %d dimensions, %d arms" % \
        (self.d, self.K)
    elif self.noise == "bernoulli":
      return "Bernoulli linear bandit: %d dimensions, %d arms" % \
        (self.d, self.K)
    elif self.noise == "beta":
      return "Beta linear bandit: %d dimensions, %d arms" % \
        (self.d, self.K)


class LogBandit(object):
  """Logistic bandit."""

  def __init__(self, X, theta):
    self.X = np.copy(X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.theta = np.copy(theta)

    self.mu = 1 / (1 + np.exp(- self.X.dot(self.theta)))
    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = (np.random.rand(self.K) < self.mu).astype(float)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Logistic bandit: %d dimensions, %d arms" % (self.d, self.K)

  @staticmethod
  def ball_env(d=3, K=10, num_env=100):
    """Arm features and theta are generated randomly in a ball."""

    env = []
    for env_id in range(num_env):
      # standard d-dimensional basis (with a bias term)
      basis = np.eye(d)
      basis[:, -1] = 1

      # arm features in a unit (d - 2)-sphere
      X = np.random.randn(K, d - 1)
      X /= np.sqrt(np.square(X).sum(axis=1))[:, np.newaxis]
      X = np.hstack((X, np.ones((K, 1))))  # bias term
      X[: basis.shape[0], :] = basis

      # parameter vector in a (d - 2)-sphere with radius 1.5
      theta = np.random.randn(d - 1)
      theta *= 1.5 / np.sqrt(np.square(theta).sum())
      theta = np.append(theta, [0])

      # create environment
      env.append(LogBandit(X, theta))
      print("%3d: %.2f %.2f | " % (env[-1].best_arm,
        env[-1].mu.min(), env[-1].mu.max()), end="")
      if (env_id + 1) % 10 == 0:
        print()

    return env


class CoBandit(object):
  """Contextual bandit with linear generalization."""

  def __init__(self, X, Theta, sigma=0.5):
    self.X = np.copy(X)  # [number of contexs] x d feature matrix
    self.Theta = np.copy(Theta)  # d x [number of arms] parameter matrix
    self.K = self.Theta.shape[1]
    self.d = self.X.shape[1]
    self.num_contexts = self.X.shape[0]
    self.sigma = sigma

    self.mu = self.X.dot(self.Theta)
    self.best_arm = np.argmax(self.mu, axis=1)

    self.randomize()

  def randomize(self):
    # choose context
    self.ct = np.random.randint(self.num_contexts)
    self.mut = self.mu[self.ct, :]

    # generate stochastic rewards
    self.rt = self.mut + self.sigma * np.random.randn(self.K)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm[self.ct]] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mut[self.best_arm[self.ct]] - self.mut[arm]

  def print(self):
    return "Contextual bandit: %d dimensions, %d arms" % (self.d, self.K)


def evaluate_one(Alg, params, env, n, period_size=1):
  """One run of a bandit algorithm."""
  alg = Alg(env, n, params)

  regret = np.zeros(n // period_size)
  for t in range(n):
    # generate state
    env.randomize()

    # take action
    arm = alg.get_arm(t)

    # update model and regret
    alg.update(t, arm, env.reward(arm))
    regret_at_t = env.regret(arm)
    regret[t // period_size] += regret_at_t

  return regret, alg


def evaluate(Alg, params, env, n=1000, period_size=1, printout=True):
  """Multiple runs of a bandit algorithm."""
  if printout:
    print("Evaluating %s" % Alg.print(), end="")
  start = time.time()

  num_exps = len(env)
  regret = np.zeros((n // period_size, num_exps))
  alg = num_exps * [None]

  dots = np.linspace(0, num_exps - 1, 100).astype(int)
  for ex in range(num_exps):
    output = evaluate_one(Alg, params, env[ex], n, period_size)
    regret[:, ex] = output[0]
    alg[ex] = output[1]

    if ex in dots:
      if printout:
        print(".", end="")
  if printout:
    print(" %.1f seconds" % (time.time() - start))

  if printout:
    total_regret = regret.sum(axis=0)
    print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)" %
      (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
      np.median(total_regret), total_regret.max(), total_regret.min()))

  return regret, alg


def evaluate_one_worker(Alg, params, exp_envs, n, period_size, shared_vars, exps):
  """One run of a bandit algorithm."""
  all_regret = shared_vars['all_regret']
  all_alg = shared_vars['all_alg']
  #ex = shared_vars['ex']
  #lock = shared_vars['lock'] 

  for exp in exps:
    env = exp_envs[exp]
    #print(exp)

    alg = Alg(env, n, params)

    regret = np.zeros(n // period_size)
    for t in range(n):
      # generate state
      env.randomize()

      # take action
      arm = alg.get_arm(t)

      # update model and regret
      alg.update(t, arm, env.reward(arm))
      regret_at_t = env.regret(arm)
      regret[t // period_size] += regret_at_t

    all_regret[:, exp] = regret
    all_alg[exp] = alg

    print(".", end="")


def evaluate_parallel(Alg, params, exp_envs, n=1000, num_process=10, 
                      period_size=1, printout=True):
  """Multiple runs of a bandit algorithm in parallel."""
  if printout:
    print("Evaluating %s" % Alg.print(), end="")
  start = time.time()

  num_exps = len(exp_envs)
  #regret = np.zeros((n // period_size, num_exps))
  #alg = num_exps * [None]

  dots = np.linspace(0, num_exps - 1, 100).astype(int)

  manager = mp.Manager()
  shared_regret = mp.Array('d', np.zeros(n // period_size * num_exps))
  all_regret = np.frombuffer(shared_regret.get_obj()).\
                reshape((n // period_size, num_exps))
  all_alg = manager.list(num_exps * [None])
  exp_dist = np.ceil(np.linspace(0, num_exps, num_process+1)).astype(int)

  #lock = manager.Lock()

  shared_vars = {'all_regret':all_regret, 'all_alg':all_alg}
  
  jobs = []
  for i in range(num_process):
    ps = mp.Process(target=evaluate_one_worker, 
          args=(Alg, params, exp_envs, n, period_size, 
                shared_vars, range(exp_dist[i], exp_dist[i+1])))
    jobs.append(ps)
    ps.start()

  for job in jobs:
    job.join()

  if printout:
    print(" %.1f seconds" % (time.time() - start))

  if printout:
    total_regret = all_regret.sum(axis=0)
    #print(total_regret)
    print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)" %
      (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
      np.median(total_regret), total_regret.max(), total_regret.min()))

  return all_regret, all_alg
