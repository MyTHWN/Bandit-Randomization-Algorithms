"""History Swapping (HS) for exploration for multi-armed and linear bandits."""

import time
import numpy as np
import copy

class HistorySwapping:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.swap_prob = 0.1

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.reward_hist = {i:[] for i in range(self.K)}
    # self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward_hist[arm].append(r)
    
    if t >= self.K and np.random.random() < self.swap_prob:
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
    if t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t
      return arm

    # decision statistics
    muhat = self.reward / self.pulls
    best_arm = np.argmax(muhat)

    # pull the arm
    arm = best_arm

    return arm

  @staticmethod
  def print():
    return "Histroy-Swapping"


class HistorySwapping_MD:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.swap_prob = 0.1

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.reward_hist = {i:[] for i in range(self.K)}
    # self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

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

  def get_arm(self, t):
    if t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t
      return arm

    # decision statistics
    muhat = self.reward / self.pulls
    best_arm = np.argmax(muhat)

    if np.random.random() < self.swap_prob:
      arm_swap = np.random.randint(self.K)
      np.random.shuffle(self.reward_hist[best_arm])
      np.random.shuffle(self.reward_hist[arm_swap])
      self.reward_hist[best_arm][0], self.reward_hist[arm_swap][0] = \
          self.reward_hist[arm_swap][0], self.reward_hist[best_arm][0]
      
      self.reward[best_arm] = np.sum(self.reward_hist[best_arm])
      self.reward[arm_swap] = np.sum(self.reward_hist[arm_swap])

      muhat = self.reward / self.pulls
      best_arm = np.argmax(muhat)

    # pull the arm
    arm = best_arm

    return arm

  @staticmethod
  def print():
    return "Histroy-Swapping_modified"


class FreshHistorySwapping(HistorySwapping):
  def get_arm(self, t):
    if t < self.K:
      # each arm is pulled once in the first K rounds
      arm = t
      return arm

    # swapped_reward_history = copy.copy(self.reward_hist)
    muhat = self.reward / self.pulls
    best_arm = np.argmax(muhat)

    if self.swap_prob > 0:
      #swap reward history between the best arm and the other arms
      swapped_reward = np.copy(self.reward)
      
      '''
      for r_idx, reward in enumerate(swapped_reward_history[best_arm]):
        if np.random.random() < self.swap_prob:
          arm_swap = np.random.randint(self.K)
          np.random.shuffle(swapped_reward_history[arm_swap])
          swapped_reward_history[best_arm][r_idx], swapped_reward_history[arm_swap][0] = \
              swapped_reward_history[arm_swap][0], swapped_reward_history[best_arm][r_idx]
      '''
    
      # num_samples = int(np.ceil(self.swap_prob*len(self.reward_hist[best_arm])))
      # sampled_rewards = np.random.choice(self.reward_hist[best_arm], num_samples)
      
      sampled_indexes = (np.random.random(len(self.reward_hist[best_arm])) < \
          self.swap_prob*np.ones(len(self.reward_hist[best_arm]))).astype(int)
      sampled_rewards = np.array(self.reward_hist[best_arm])[np.where(sampled_indexes == 1)[0]]

      swapped_reward[best_arm] -= np.sum(sampled_rewards)

      arms_to_swap = np.random.choice(self.K, len(sampled_rewards), replace=True)
      for i, reward in enumerate(sampled_rewards):
        arm_swap = arms_to_swap[i]
        sample_a_reward = np.random.choice(self.reward_hist[arm_swap])
        # sample_a_reward = self.reward_hist[arm_swap][np.random.randint(len(self.reward_hist[arm_swap]))]
        swapped_reward[arm_swap] += - sample_a_reward + reward
        swapped_reward[best_arm] += sample_a_reward

      muhat = swapped_reward / self.pulls
      best_arm = np.argmax(muhat)
    
    arm = best_arm

    return arm

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.reward_hist[arm].append(r)

  @staticmethod
  def print():
    return "Fresh Histroy-Swapping"


class LinHistorySwap:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    #self.swap_prob = swap_prob

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
    self.reward_hist[arm].append(r)
    
    if t >= self.K and np.random.random() < self.swap_prob:# * np.sqrt(self.K / (t + 1)) / 2:
      #arm_swap = np.random.randint(self.K)
      if self.swap_top == -1:
        arm_swap = np.random.randint(self.K)
      else:
        arm_swap = np.random.choice(self.mu.argsort()[-self.swap_top:])
      np.random.shuffle(self.reward_hist[arm])
      np.random.shuffle(self.reward_hist[arm_swap])
      self.reward_hist[arm][0], self.reward_hist[arm_swap][0] = \
        self.reward_hist[arm_swap][0], self.reward_hist[arm][0]
      
      self.reward[arm] = np.sum(self.reward_hist[arm])
      self.reward[arm_swap] = np.sum(self.reward_hist[arm_swap])
    
    else: 
      self.reward[arm] += r #np.sum[self.reward_hist[arm]]

  def get_arm(self, t):
    if t < self.K:
      arm = t
      return arm

    Gram = np.tensordot(self.pulls, self.X2, \
      axes=([0], [0]))
    B = self.X.T.dot(self.reward)

    reg = 3e-3 * np.eye(self.d)
    # Gram_inv = np.linalg.inv(Gram + reg)
    # theta = Gram_inv.dot(B)
    theta = np.linalg.solve(Gram + reg, B)
    self.mu = self.X.dot(theta) + 1e-6 * np.random.rand(self.K)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinHistory-Swapping"


class AllHistorySwapping(HistorySwapping):
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
        sampled_indexes = (np.random.random(len(self.reward_hist[arm])) \
      		< self.swap_prob*np.ones(len(self.reward_hist[arm]))).astype(int)
        sampled_rewards = np.array(self.reward_hist[arm])\
      			  [np.where(sampled_indexes == 1)[0]]
        #num_samples = int(np.ceil(self.swap_prob * len(self.reward_hist[arm])))
        #sampled_rewards = np.random.choice(self.reward_hist[arm], num_samples)
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

  @staticmethod
  def print():
    return "All Histroy-Swapping"


class LinFreshHistorySwap(LinHistorySwap):
  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r #np.sum[self.reward_hist[arm]]
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
      #swap reward history between the best arm and the other arms
      swapped_reward = np.copy(self.reward)
    
      num_samples = int(np.ceil(self.swap_prob*len(self.reward_hist[best_arm])))
      sampled_rewards = np.random.choice(self.reward_hist[best_arm], num_samples)
      swapped_reward[best_arm] -= np.sum(sampled_rewards)

      arms_to_swap = np.random.choice(self.K, len(sampled_rewards), replace=True)
      for i, reward in enumerate(sampled_rewards):
        arm_swap = arms_to_swap[i]
        # sample_a_reward = np.random.choice(self.reward_hist[arm_swap])
        sample_a_reward = self.reward_hist[arm_swap][np.random.randint(len(self.reward_hist[arm_swap]))]
        swapped_reward[arm_swap] += - sample_a_reward + reward
        swapped_reward[best_arm] += sample_a_reward

      swapped_B = self.X.T.dot(swapped_reward)
      swapped_theta = np.linalg.solve(Gram + reg, swapped_B)
      swapped_mu = self.X.dot(swapped_theta) + 1e-6 * np.random.rand(self.K)
      
      best_arm = np.argmax(swapped_mu)

    arm = best_arm
    return arm

  @staticmethod
  def print():
    return "Lin Fresh History-Swapping"
    
    
class LinAllHistorySwap(LinHistorySwap):
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
        sampled_indexes = (np.random.random(len(self.reward_hist[arm])) \
      		< self.swap_prob*np.ones(len(self.reward_hist[arm]))).astype(int)
        sampled_rewards = np.array(self.reward_hist[arm])\
      			  [np.where(sampled_indexes == 1)[0]]
        # num_samples = int(np.ceil(self.swap_prob * len(self.reward_hist[arm])))
        # sampled_rewards = np.random.choice(self.reward_hist[arm], num_samples)
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
    return "Lin All History-Swapping"



