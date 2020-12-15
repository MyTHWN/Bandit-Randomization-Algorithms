"""History Swapping (HS) for exploration for multi-armed and linear bandits."""

import time
import numpy as np
import copy


class HS_SWR_scale:
  def __init__(self, K, n): #, params):
    self.K = K
    self.sample_portion = 0.6
    self.z = 0.6

    # for attr, val in params.items():
    #   setattr(self, attr, val)

    self.init_pulls = 2*np.log(n) / (self.z-1-np.log(self.z))
    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.ones(self.K)  # cumulative reward
    self.all_rewards = []
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

  def get_action(self, t, num_pulls):
    if t < np.ceil(self.init_pulls / float(num_pulls)):
            # or t < np.ceil(self.K / float(num_pulls)):
      # each arm is pulled at least once in the first few rounds
      action = [item % self.K for item in
                range(t * num_pulls, (t+1) * num_pulls)]
      return action

    if self.sample_portion > 0:
      # Mixing reward history among arms
      swapped_reward = np.copy(self.reward)
      swapped_pulls = np.copy(self.pulls)
      reward_pool = np.copy(self.all_rewards)

      for arm in range(self.K):
        sampled_indexes = np.random.randint(len(reward_pool),
                                            size=int(self.pulls[arm]))
        sampled_rewards = np.array(reward_pool)[sampled_indexes]
        swapped_reward[arm] += self.sample_portion * np.sum(sampled_rewards)
        # swapped_pulls[arm] += num_samples

      muhat = swapped_reward / swapped_pulls + self.tiebreak
      action = np.argsort(muhat)[::-1][:num_pulls]

    else:
      muhat = self.reward / self.pulls + self.tiebreak
      action = np.argsort(muhat)[::-1][:num_pulls]

    return action

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      # non_examined_action = action[last_click+1:]
      action = action[:last_click+1]
      r = r[:last_click+1]

      # for item in non_examined_action:
      #   if self.pulls[item] == 0:
      #     self.pulls[item] += 1
      #     self.reward[item] += 0.5
      #     self.all_rewards.append(0.5)

    self.pulls[action] += 1
    self.reward[action] += r
    self.all_rewards += list(r)

  @staticmethod
  def print():
    return "HS-SampleWithReplacement"


class PHE:
  def __init__(self, K, n):
    self.K = K
    self.a = 0.5

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.ones(self.K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[:last_click+1]
      r = r[:last_click+1]

    self.pulls[action] += 1
    self.reward[action] += r

  def get_action(self, t, num_pulls):
    self.mu = np.zeros(self.K)

    # history perturbation
    pseudo_pulls = np.ceil(self.a * self.pulls).astype(int)
    pseudo_reward = np.random.binomial(pseudo_pulls, 0.5)
    # pseudo_reward = np.random.normal(0.5 * pseudo_pulls, \
    #                                  0.5 * np.sqrt(pseudo_pulls))
    self.mu = (self.reward + pseudo_reward) / \
      (self.pulls + pseudo_pulls) + self.tiebreak

    action = np.argsort(self.mu)[::-1][:num_pulls]
    return action

  @staticmethod
  def print():
    return "PHE"


class TS:
  def __init__(self, K, n):
    self.K = K
    self.crs = 1.0  # confidence region scaling

    self.alpha = np.ones(self.K)  # positive observations
    self.beta = np.ones(self.K)  # negative observations

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[:last_click+1]
      r = r[:last_click+1]

    self.alpha[action] += r
    self.beta[action] += 1 - r

  def get_action(self, t, num_pulls):
    # posterior sampling
    crs2 = np.square(self.crs)
    self.mu = np.random.beta(self.alpha / crs2, self.beta / crs2)

    action = np.argsort(self.mu)[::-1][:num_pulls]
    return action

  @staticmethod
  def print():
    return "TS"


class CascadeUCB1:
  def __init__(self, K, T):
    self.K = K

    self.pulls = 1e-6 * np.ones(K)  # number of pulls
    self.reward = 1e-6 * np.random.rand(K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(K)  # tie breaking

    self.ucb = np.zeros(K)

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    self.pulls[action] += 1
    self.reward[action] += r

    # UCBs
    t += 1  # time starts at one
    ct = np.maximum(np.sqrt(2 * np.log(t)), 2)
    self.ucb = self.reward / self.pulls + ct * np.sqrt(1 / self.pulls)

  def get_action(self, t, num_pulls):
    action = np.argsort(self.ucb + self.tiebreak)[:: -1][: num_pulls]
    return action


class CascadeKLUCB:
  def __init__(self, K, T):
    self.K = K

    self.pulls = 1e-6 * np.ones(K)  # number of pulls
    self.reward = 1e-6 * np.random.rand(K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(K)  # tie breaking

    self.ucb = (1 - 1e-6) * np.ones(K)

  def UCB(self, p, N, t):
    C = (np.log(t) + 3 * np.log(np.log(t) + 1)) / N
    tol = 1e-5

    kl = p * np.log(p / self.ucb) + (1 - p) * np.log((1 - p) / (1 - self.ucb))
    for k in np.flatnonzero(np.abs(kl - C) > tol):
      ql = min(max(p[k], 1e-6), 1 - 1e-6)
      qu = 1 - 1e-6
      while qu - ql > tol:
        q = (ql + qu) / 2
        f = p[k] * np.log(p[k] / q) + (1 - p[k]) * np.log((1 - p[k]) / (1 - q))
        if f < C[k]:
          ql = q
        else:
          qu = q
      self.ucb[k] = qu

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    self.pulls[action] += 1
    self.reward[action] += r

    # UCBs
    t += 1  # time starts at one
    self.UCB(self.reward / self.pulls, self.pulls, t)

  def get_action(self, t, num_pulls):
    action = np.argsort(self.ucb + self.tiebreak)[:: -1][: num_pulls]
    return action


class TopRank:
  def __init__(self, K, T):
    self.K = K
    self.T = T

    self.pulls = np.ones((K, K))  # number of pulls
    self.reward = np.zeros((K, K))  # cumulative reward

    self.G = np.ones((K, K), dtype=bool)
    self.P = np.zeros(K)
    self.P2 = np.ones((K, K))

  def rerank(self):
    Gt = (self.reward / self.pulls - 2 * np.sqrt(np.log(self.T) / self.pulls)) > 0
    if not np.array_equal(Gt, self.G):
      self.G = np.copy(Gt)

      Pid = 0
      self.P = - np.ones(self.K)
      while (self.P == -1).sum() > 0:
        items = np.flatnonzero(Gt.sum(axis=0) == 0)
        self.P[items] = Pid
        Gt[items, :] = 0
        Gt[:, items] = 1
        Pid += 1

      self.P2 = \
        (np.tile(self.P[np.newaxis], (self.K, 1)) == np.tile(self.P[np.newaxis].T, (1, self.K))).astype(float)

  def update(self, t, action, r):
    clicks = np.zeros(self.K)
    clicks[action] = r

    M = np.outer(clicks, 1 - clicks) * self.P2
    self.pulls += M + M.T
    self.reward += M - M.T

    self.rerank()

  def get_action(self, t, num_pulls):
    action = np.argsort(self.P + 1e-6 * np.random.rand(self.K))[: num_pulls]
    return action