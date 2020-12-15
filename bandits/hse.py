"""History Swapping (HS) for exploration for multi-armed and linear bandits."""

import time
import numpy as np
import copy


class HistorySwapping:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.swap_prob = 0.15
    self.sample_method = 'random'

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.reward_hist = {i:[] for i in range(self.K)}

    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def get_arm(self, t):
    if t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t
      return arm

    # swapped_reward_history = copy.copy(self.reward_hist)
    muhat = self.reward / self.pulls
    best_arm = np.argmax(muhat)

    if self.swap_prob > 0:
      # swap reward history between the best arm and the other arms
      swapped_reward = np.copy(self.reward)
      reward_pool = []
      reward_nums = []

      for arm in range(self.K):
        if self.sample_method == 'random':
          sampled_indexes = (np.random.random(len(self.reward_hist[arm])) \
        		< self.swap_prob*np.ones(len(self.reward_hist[arm]))).astype(int)
          sampled_rewards = np.array(self.reward_hist[arm])\
        			  [np.where(sampled_indexes == 1)[0]]
        elif self.sample_method == 'ceil':
          num_samples = int(np.ceil(self.swap_prob * len(self.reward_hist[arm])))
          sampled_rewards = np.random.choice(self.reward_hist[arm], num_samples)
        
        swapped_reward[arm] -= np.sum(sampled_rewards)
        reward_pool += list(sampled_rewards)
        num_samples = len(list(sampled_rewards))
        reward_nums.append(num_samples)

      np.random.shuffle(reward_pool)
      reward_pool_pointer = 0
      for arm in range(self.K):
        swapped_reward[arm] += np.sum(reward_pool[reward_pool_pointer: \
        			reward_pool_pointer+reward_nums[arm]])
        reward_pool_pointer += reward_nums[arm]

      muhat = swapped_reward / self.pulls
      best_arm = np.argmax(muhat)

    arm = best_arm

    return arm

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.reward_hist[arm].append(r)

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  @staticmethod
  def print():
    return "Histroy-Swapping"


class LinHistorySwap:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.swap_prob = 0.15
    self.sample_method = 'random'

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pulls = np.zeros(self.K, dtype=int) # number of pulls
    self.reward = np.zeros(self.K) # cumulative reward
    #self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
    self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
    for k in range(self.K):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])
    self.reward_hist = {i:[] for i in range(self.K)}

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r  # np.sum[self.reward_hist[arm]]
    self.reward_hist[arm].append(r)

  def get_arm(self, t):
    if t < self.K:
      arm = t
      return arm

    Gram = np.tensordot(self.pulls, self.X2, \
                        axes=([0], [0]))
    B = self.X.T.dot(self.reward)

    reg = 1e-3 * np.eye(self.d)
    # Gram_inv = np.linalg.inv(Gram + reg)
    # theta = Gram_inv.dot(B)
    theta = np.linalg.solve(Gram + reg, B)
    self.mu = self.X.dot(theta) + 1e-6 * np.random.rand(self.K)

    best_arm = np.argmax(self.mu)

    if self.swap_prob > 0:
      # swap reward history between the best arm and the other arms
      swapped_reward = np.copy(self.reward)
      reward_pool = []
      reward_nums = []

      for arm in range(self.K):
        if self.sample_method == 'random':
          sampled_indexes = (np.random.random(len(self.reward_hist[arm])) \
        		< self.swap_prob*np.ones(len(self.reward_hist[arm]))).astype(int)
          sampled_rewards = np.array(self.reward_hist[arm])\
        		[np.where(sampled_indexes == 1)[0]]
        elif self.sample_method == 'ceil':
          num_samples = int(np.ceil(self.swap_prob * len(self.reward_hist[arm])))
          sampled_rewards = np.random.choice(self.reward_hist[arm], num_samples)
        
        swapped_reward[arm] -= np.sum(sampled_rewards)
        reward_pool += list(sampled_rewards)
        num_samples = len(list(sampled_rewards))
        reward_nums.append(num_samples)

      np.random.shuffle(reward_pool)
      reward_pool_pointer = 0
      for arm in range(self.K):
        swapped_reward[arm] += np.sum(reward_pool[reward_pool_pointer: \
                                  reward_pool_pointer+reward_nums[arm]])
        reward_pool_pointer += reward_nums[arm]

      swapped_B = self.X.T.dot(swapped_reward)
      swapped_theta = np.linalg.solve(Gram + reg, swapped_B)
      swapped_mu = self.X.dot(swapped_theta) + 1e-6 * np.random.rand(self.K)

      best_arm = np.argmax(swapped_mu)

    arm = best_arm
    return arm

  @staticmethod
  def print():
    return "Lin History-Swapping"


class HS_SWR:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.sample_portion = 1
    self.z = 0.5

    for attr, val in params.items():
      setattr(self, attr, val)

    self.init_pulls = 2*np.log(n) / (self.z-1-np.log(self.z))
    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.all_rewards = []
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def get_arm(self, t):
    # if t < self.K:
    if t < self.init_pulls or t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t % self.K
      return arm

    if self.sample_portion > 0:
      # Mixing reward history among arms
      swapped_reward = np.copy(self.reward)
      swapped_pulls = np.copy(self.pulls)
      reward_pool = np.copy(self.all_rewards)

      for arm in range(self.K):
        num_samples = int(np.ceil(self.sample_portion * self.pulls[arm]))
        # sampled_rewards = np.random.choice(reward_pool, num_samples,
        #                                    replace=True)
        sampled_indexes = np.random.randint(len(reward_pool),
                                            size=num_samples)
        sampled_rewards = np.array(reward_pool)[sampled_indexes]
        swapped_reward[arm] += np.sum(sampled_rewards)
        # swapped_pulls[arm] += num_samples

      muhat = swapped_reward / swapped_pulls + self.tiebreak
      best_arm = np.argmax(muhat)

    else:
      muhat = self.reward / self.pulls + self.tiebreak
      best_arm = np.argmax(muhat)

    arm = best_arm
    return arm

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.all_rewards.append(r)

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  @staticmethod
  def print():
    return "HS-SampleWithReplacement"


class HS_SWR_scale:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.sample_portion = 1
    self.z = 0.5

    for attr, val in params.items():
      setattr(self, attr, val)

    self.init_pulls = 2*np.log(n) / (self.z-1-np.log(self.z)) + 1
    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.all_rewards = []
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking

    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def get_arm(self, t):
    # if t < self.K:
    if t < self.init_pulls or t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t % self.K
      return arm

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
      best_arm = np.argmax(muhat)

    else:
      muhat = self.reward / self.pulls + self.tiebreak
      best_arm = np.argmax(muhat)

    arm = best_arm
    return arm

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.all_rewards.append(r)

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  @staticmethod
  def print():
    return "HS-SampleWithReplacement"


class LinHS_SWR_scale:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.sample_portion = 1
    self.z = 0.6

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pulls = np.zeros(self.K, dtype=int)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
    self.X2 = np.zeros((self.K, self.d, self.d))
    # outer products of arm features
    for k in range(self.K):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])
    self.all_rewards = []

    self.init_pulls = 2 * np.log(n) / (self.z - 1 - np.log(self.z)) +1

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.all_rewards.append(r)

  def get_arm(self, t):

    if t < self.init_pulls or t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t % self.K
      return arm

    if self.sample_portion > 0:
      swapped_reward = np.copy(self.reward)
      swapped_pulls = np.copy(self.pulls)
      mean_all_rewards = np.mean(self.all_rewards)
      reward_pool = np.copy(self.all_rewards)

      for arm in range(self.K):

        sampled_indexes = np.random.randint(len(reward_pool),
                                            size=self.pulls[arm])
        sampled_rewards = np.array(reward_pool)[sampled_indexes]
        swapped_reward[arm] += self.sample_portion * (np.sum(sampled_rewards) - \
                               self.pulls[arm] * mean_all_rewards)
        # swapped_pulls[arm] += num_samples

      swapped_Gram = np.tensordot(swapped_pulls, self.X2, \
                          axes=([0], [0]))
      swapped_B = self.X.T.dot(swapped_reward)
      reg = 1e-3 * np.eye(self.d)
      swapped_theta = np.linalg.solve(swapped_Gram + reg, swapped_B)
      swapped_mu = self.X.dot(swapped_theta) + self.tiebreak

      best_arm = np.argmax(swapped_mu)

    else:
      Gram = np.tensordot(self.pulls, self.X2, \
                          axes=([0], [0]))
      B = self.X.T.dot(self.reward)
      reg = 1e-3 * np.eye(self.d)
      # Gram_inv = np.linalg.inv(Gram + reg)
      # theta = Gram_inv.dot(B)
      theta = np.linalg.solve(Gram + reg, B)
      self.mu = self.X.dot(theta) + self.tiebreak

      best_arm = np.argmax(self.mu)

    arm = best_arm
    return arm

  @staticmethod
  def print():
    return "Lin HS-SWR-scale"


class LinHS_SWR:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.sample_portion = 1

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pulls = np.zeros(self.K, dtype=int)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K) # tie breaking
    self.X2 = np.zeros((self.K, self.d, self.d))
    # outer products of arm features
    for k in range(self.K):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])
    self.all_rewards = []

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.all_rewards.append(r)

  def get_arm(self, t):
    if t < self.K:
      arm = t
      return arm

    if self.sample_portion > 0:
      swapped_reward = np.copy(self.reward)
      swapped_pulls = np.copy(self.pulls)
      # np.random.shuffle(self.all_rewards)
      mean_all_rewards = np.mean(self.all_rewards)
      reward_pool = np.copy(self.all_rewards)

      for arm in range(self.K):
        num_samples = int(np.ceil(self.sample_portion * self.pulls[arm]))
        # sampled_rewards = np.random.choice(reward_pool, num_samples,
        #                                    replace=True)
        sampled_indexes = np.random.randint(len(reward_pool),
                                            size=num_samples)
        sampled_rewards = np.array(reward_pool)[sampled_indexes]
        swapped_reward[arm] += np.sum(sampled_rewards) - \
                               num_samples * mean_all_rewards
        # swapped_pulls[arm] += num_samples

      swapped_Gram = np.tensordot(swapped_pulls, self.X2, \
                          axes=([0], [0]))
      swapped_B = self.X.T.dot(swapped_reward)
      reg = 1e-3 * np.eye(self.d)
      swapped_theta = np.linalg.solve(swapped_Gram + reg, swapped_B)
      swapped_mu = self.X.dot(swapped_theta) + self.tiebreak

      best_arm = np.argmax(swapped_mu)

    else:
      Gram = np.tensordot(self.pulls, self.X2, \
                          axes=([0], [0]))
      B = self.X.T.dot(self.reward)
      reg = 1e-3 * np.eye(self.d)
      # Gram_inv = np.linalg.inv(Gram + reg)
      # theta = Gram_inv.dot(B)
      theta = np.linalg.solve(Gram + reg, B)
      self.mu = self.X.dot(theta) + self.tiebreak

      best_arm = np.argmax(self.mu)

    arm = best_arm
    return arm

  @staticmethod
  def print():
    return "Lin HS-SampleWithReplacement"


class HS_SWR_MirrorPool(HS_SWR):

  def get_arm(self, t):
    if t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t
      return arm

    if self.sample_portion > 0:
      # Mixing reward history among arms
      swapped_reward = np.copy(self.reward)
      swapped_pulls = np.copy(self.pulls)
      mean_all_rewards = np.mean(self.all_rewards)
      mirrors = 2 * mean_all_rewards - np.array(self.all_rewards)
      mirror_reward_pool = np.concatenate((mirrors, self.all_rewards))

      for arm in range(self.K):
        num_samples = int(np.ceil(self.sample_portion * self.pulls[arm]))
        # sampled_rewards = np.random.choice(self.all_rewards, num_samples,
        #                                    replace=True)
        sampled_indexes = np.random.randint(len(mirror_reward_pool),
                                            size=num_samples)
        sampled_rewards = np.array(mirror_reward_pool)[sampled_indexes]
        swapped_reward[arm] += np.sum(sampled_rewards)
        # swapped_pulls[arm] += num_samples

      muhat = swapped_reward / swapped_pulls + self.tiebreak
      best_arm = np.argmax(muhat)

    else:
      muhat = self.reward / self.pulls + self.tiebreak
      best_arm = np.argmax(muhat)

    arm = best_arm
    return arm

  @staticmethod
  def print():
    return "HS-SWR_mirrored_pool"


class LinHS_SWR_MirrorPool(LinHS_SWR):

  def get_arm(self, t):
    if t < self.K:
      arm = t
      return arm

    if self.sample_portion > 0:
      swapped_reward = np.copy(self.reward)
      swapped_pulls = np.copy(self.pulls)
      # np.random.shuffle(self.all_rewards)
      mean_all_rewards = np.mean(self.all_rewards)
      mirrors = 2 * mean_all_rewards - np.array(self.all_rewards)
      mirror_reward_pool = np.concatenate((mirrors, self.all_rewards))

      for arm in range(self.K):
        num_samples = int(np.ceil(self.sample_portion * self.pulls[arm]))
        # sampled_rewards = np.random.choice(mirror_reward_pool, num_samples,
        #                                    replace=True)
        sampled_indexes = np.random.randint(len(mirror_reward_pool),
                                            size=num_samples)
        sampled_rewards = np.array(mirror_reward_pool)[sampled_indexes]
        swapped_reward[arm] += np.sum(sampled_rewards) - \
                               num_samples * mean_all_rewards
        # swapped_pulls[arm] += num_samples

      swapped_Gram = np.tensordot(swapped_pulls, self.X2, \
                          axes=([0], [0]))
      swapped_B = self.X.T.dot(swapped_reward)
      reg = 1e-3 * np.eye(self.d)
      swapped_theta = np.linalg.solve(swapped_Gram + reg, swapped_B)
      swapped_mu = self.X.dot(swapped_theta) + self.tiebreak

      best_arm = np.argmax(swapped_mu)

    else:
      Gram = np.tensordot(self.pulls, self.X2, \
                          axes=([0], [0]))
      B = self.X.T.dot(self.reward)

      reg = 1e-3 * np.eye(self.d)
      # Gram_inv = np.linalg.inv(Gram + reg)
      # theta = Gram_inv.dot(B)
      theta = np.linalg.solve(Gram + reg, B)
      self.mu = self.X.dot(theta) + self.tiebreak

      best_arm = np.argmax(self.mu)

    arm = best_arm
    return arm

  @staticmethod
  def print():
    return "Lin HS-SWR_mirrored_pool"