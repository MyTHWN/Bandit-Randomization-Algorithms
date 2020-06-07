"""Deep environments and bandit algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import cPickle
import numpy as np
# from scipy.spatial.distance import cdist
from tensorflow import keras

from google3.pyglib import gfile


class BinaryClassBandit(object):
  """Binary classification bandit."""

  def __init__(self, X, y, pos_label=1, K=2):
    # self.pos = np.flatnonzero(y == pos_label)
    # self.neg = np.flatnonzero(y != pos_label)
    self.Xall = X
    self.y = np.copy(y)

    self.K = K
    self.n = self.Xall.shape[0]
    self.d = self.Xall.shape[1]
    # self.p1 = 1

    # self.y = y == pos_label
    # self.p1 = 1 - np.power(1 - self.y.mean(), self.K)

#     self.py = np.zeros(self.n)
#     batch = 10000
#     for first_ndx in range(0, self.n, batch):
#       if first_ndx < self.n:
#         last_ndx = min(first_ndx + batch, self.n)
#         D = cdist(self.Xall[first_ndx : last_ndx, :], self.Xall)
#         ynn = (y == pos_label)[np.argsort(D, axis=1)[:, : 10]]
#         self.py[first_ndx : last_ndx] = ynn.mean(axis=1)
#
#     print(self.py[np.random.randint(self.n, size=200)])

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.ndx = np.random.randint(self.n, size=self.K)
    self.X = self.Xall[self.ndx, :]
    self.rt = np.random.rand(self.K) < self.y[self.ndx]

    # neg_ndx = self.neg[np.random.randint(self.neg.size, size=self.K)]
    # self.X = self.Xall[neg_ndx, :]
    # self.rt = np.random.rand(self.K) < 0.25

    # pos_ndx = self.pos[np.random.randint(self.pos.size, size=self.K // 2)]
    # self.X[- self.K // 2 :, :] = self.Xall[pos_ndx, :]
    # self.rt[- self.K // 2 :] = np.random.rand(self.K // 2) < 0.75

    # pos_arm = np.random.randint(self.K)
    # pos_ndx = self.pos[np.random.randint(self.pos.size)]
    # self.X[pos_arm, :] = self.Xall[pos_ndx, :]
    # self.rt[pos_arm] = 1

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous REWARD of the arm
    return self.rt[arm]

  def print(self):
    return "Binary classification bandit"


def load_dataset(dataset, maxd):
  data_dir = "/cns/pw-d/home/bkveton/PHE/Datasets"

  print("Preprocessing dataset %s..." % dataset)
  if dataset == "cifar-10":
    for num in range(6):
      with gfile.Open("%s/%s-%d" % (data_dir, dataset, num + 1)) as f:
        data = cPickle.load(f)
      if not num:
        X = np.asarray(data["data"]) / 255
        y = np.asarray(data["labels"])
      else:
        X = np.vstack((X, np.asarray(data["data"]) / 255))
        y = np.concatenate((y, np.asarray(data["labels"])))
  if dataset == "digit_recognition":
    with gfile.Open("%s/%s.txt" % (data_dir, dataset)) as f:
      D = np.loadtxt(f, delimiter=",")
    X = D[:, : -1] / 16
    y = D[:, -1]
  if dataset == "iris":
    with gfile.Open("%s/%s.txt" % (data_dir, dataset)) as f:
      D = np.loadtxt(f, delimiter=",")
    X = D[:, : -1]
    y = D[:, -1]
  elif dataset == "letter_recognition":
    with gfile.Open("%s/%s.txt" % (data_dir, dataset)) as f:
      D = np.genfromtxt(f, dtype="str", delimiter=",")
    X = D[:, 1 :].astype("float")
    y = np.asarray(map(ord, D[:, 0]))
    y = y - y.min()
  elif dataset == "mnist":
    with gfile.Open("%s/%s.npz" % (data_dir, dataset)) as f:
      D = np.load(f)
      X = np.vstack((np.reshape(D["x_train"], (60000, -1)),
        np.reshape(D["x_test"], (10000, -1)))).astype(float)
      y = np.concatenate((D["y_train"], D["y_test"]))

  X = X - X.mean(axis=0)[np.newaxis, :]  # center data

  # random projections
  if X.shape[1] > maxd:
    print("%d features are projected on %d random vectors." %
      (X.shape[1], maxd))
    U = np.random.randn(X.shape[1], maxd) / np.sqrt(X.shape[1])
    X = X.dot(U)

  X = np.hstack((X, np.ones((y.size, 1))))  # add bias term
  print("%d examples, %d features, %d labels" %
    (X.shape[0], X.shape[1], y.max() + 1))

  return X, y


class DeepFPL(object):
  """Deep follow the perturbed leader."""

  def __init__(self, env, n, params):
    self.env = env
    self.n = n
    self.K = self.env.X.shape[0]
    self.d = self.env.X.shape[1]

    self.hidden_nodes = 0  # number of nodes in the hidden layer

    self.a = 0.5  # perturbation scale (Gaussian noise)
    self.num_models = 3  # ensemble size
    self.lr = 1.0 / np.sqrt(self.n)  # learning rate
    self.batch_size = 32  # mini-batch size
    self.init_pulls = self.d  # number of initial random pulls

    for attr, val in params.iteritems():
      setattr(self, attr, val)

    # sufficient statistics
    self.model_size = self.n  # model size
    self.X = np.zeros((self.model_size, self.d))
    self.y = self.a * np.minimum(np.maximum(
      np.random.randn(self.num_models, self.model_size), -6), 6)

    # neural net
    self.num_layers = 2 if self.hidden_nodes else 1
    self.model = self.num_models * [None]
    for h in range(self.num_models):
      self.model[h] = keras.Sequential()
      if self.num_layers == 1:
        self.model[h].add(keras.layers.Dense(1,
          activation="sigmoid", input_shape=(self.d,), use_bias=False))
      else:
        self.model[h].add(keras.layers.Dense(self.hidden_nodes,
          activation="tanh", input_shape=(self.d,), use_bias=False))
        self.model[h].add(keras.layers.Dense(1, activation="sigmoid"))
      self.model[h].compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(lr=self.lr))

  def update(self, t, arm, r):
    self.X[t, :] = self.env.X[arm, :]
    for h in range(self.num_models):
      self.y[h, t] += r

    if t == self.n - 1:
      keras.backend.clear_session()

  def get_arm(self, t):
    if t:
      # sub = np.random.randint(t, size=self.batch_size)
      sub = np.arange(max(t - self.batch_size, 0), t)
      for h in range(self.num_models):
        self.model[h].train_on_batch(self.X[sub, :], self.y[h, sub])

    self.mu = np.zeros(self.K)
    if t < self.init_pulls:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      h = np.random.randint(self.num_models)
      self.mu = self.model[h].predict(self.env.X).flatten()

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "DeepFPL"


class DeepFL(DeepFPL):
  """Deep follow the leader."""

  def __init__(self, env, n, params):
    params["a"] = 0
    params["num_models"] = 1
    DeepFPL.__init__(self, env, n, params)

  @staticmethod
  def print():
    return "DeepFL"


class DeepPHE(object):
  """Deep perturbed-history exploration."""

  def __init__(self, env, n, params):
    self.env = env
    self.n = n
    self.K = self.env.X.shape[0]
    self.d = self.env.X.shape[1]

    self.hidden_nodes = 0  # number of nodes in the hidden layer

    self.a = 1  # number of pseudo-rewards per observed reward
    self.num_models = 3  # ensemble size
    self.lr = 1.0 / np.sqrt(self.n)  # learning rate
    self.batch_size = 128  # mini-batch size
    self.init_pulls = self.d  # number of initial random pulls

    for attr, val in params.iteritems():
      setattr(self, attr, val)

    # sufficient statistics
    self.model_size = (self.a + 1) * self.n  # model size
    self.X = np.zeros((self.num_models, self.model_size, self.d))
    self.y = np.random.randint(2,
      size=(self.num_models, self.model_size)).astype(float)

    # neural net
    self.num_layers = 2 if self.hidden_nodes else 1
    self.model = self.num_models * [None]
    for h in range(self.num_models):
      self.model[h] = keras.Sequential()
      if self.num_layers == 1:
        self.model[h].add(keras.layers.Dense(1,
          activation="sigmoid", input_shape=(self.d,)))
      else:
        self.model[h].add(keras.layers.Dense(self.hidden_nodes,
          activation="tanh", input_shape=(self.d,)))
        self.model[h].add(keras.layers.Dense(1, activation="sigmoid"))
      self.model[h].compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(lr=self.lr))

  def update(self, t, arm, r):
    for h in range(self.num_models):
      self.y[h, (self.a + 1) * t] = r
      for j in range(self.a + 1):
        self.X[h, (self.a + 1) * t + j, :] = self.env.X[arm, :]

    if t == self.n - 1:
      keras.backend.clear_session()

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.init_pulls:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      sub = np.random.randint(t, size = self.batch_size)
      sub = ((self.a + 1) * sub[:, np.newaxis] +
        np.arange(self.a + 1)[np.newaxis, :]).flatten()

      for h in range(self.num_models):
        self.model[h].train_on_batch(self.X[h, sub, :], self.y[h, sub])
        # self.model[h].fit(self.X[h, sub, :], self.y[h, sub],
        #   batch_size=32, epochs=1, verbose=0)
      h = np.random.randint(self.num_models)
      self.mu = self.model[h].predict(self.env.X).flatten()

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "DeepPHE"
