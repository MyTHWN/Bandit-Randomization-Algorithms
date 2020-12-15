
import time
import numpy as np
import multiprocessing as mp

class SpikeEnv:
  def __init__(self, K = 4, L = 16, baseu = 0.5, gapu = 0.4, basev = 0.5, gapv = 0.4):
    self.K = K
    self.L = L

    self.ubar = baseu * np.ones(K)
    self.ubar[0] += gapu
    self.vbar = basev * np.ones(L)
    self.vbar[L // 2] += gapv
    self.best_action = np.argsort(self.vbar)[: : -1][: self.K]

    self.ut = np.zeros(K)
    self.vt = np.zeros(L)

  def randomize(self):
    # sample random variables
    self.ut = np.array(np.random.rand(self.K) < self.ubar, dtype = np.int)
    self.vt = np.array(np.random.rand(self.L) < self.vbar, dtype = np.int)

  def reward(self, action):
    # reward of action (chosen items)
    return np.multiply(self.ut, self.vt[action])

  def regret(self, action):
    # regret of action (chosen items)
    return np.dot(self.ut, self.vt[self.best_action]) - np.dot(self.ut, self.vt[action])

  def pregret(self, action):
    # expected regret of action (chosen items)
    return np.dot(self.ubar, self.vbar[self.best_action]) - np.dot(self.ubar, self.vbar[action])

  def plot(self):
    # plot model parameters
    fig, (left, right) = plt.subplots(ncols = 2, figsize = (10, 3))
    left.plot(self.ubar)
    right.plot(self.vbar)
    plt.show()

class PBMEnv(SpikeEnv):
  def __init__(self, filename):
    self.K = 5
    self.L = 10

    self.ubar = np.zeros(self.K)
    self.vbar = np.zeros(self.L)

    with open(filename) as f:
      f.readline()
      vals = f.readline().split(",");
      for i in range(self.K):
        self.ubar[i] = float(vals[i])
      vals = f.readline().split(",");
      for i in range(self.L):
        self.vbar[i] = float(vals[i])

    self.best_action = np.argsort(self.vbar)[: : -1][: self.K]

    self.ut = np.zeros(self.K)
    self.vt = np.zeros(self.L)

class CMEnv:
  def __init__(self, filename):
    self.K = 5
    self.L = 10

    self.vbar = np.zeros(self.L)

    with open(filename) as f:
      f.readline()
      vals = f.readline().split(",");
      for i in range(self.L):
        self.vbar[i] = float(vals[i])

    self.best_action = np.argsort(self.vbar)[: : -1][: self.K]

    self.vt = np.zeros(self.L)

  def randomize(self):
    # sample random variables
    self.vt = np.array(np.random.rand(self.L) < self.vbar, dtype = np.int)

  def reward(self, action):
    # reward of action (chosen items)
    r = self.vt[action]
    if r.sum() > 0:
      first_click = np.flatnonzero(r)[0]
      r[first_click + 1 :] = 0
    return r

  def regret(self, action):
    # regret of action (chosen items)
    return np.prod(1 - self.vt[action]) - np.prod(1 - self.vt[self.best_action])

  def pregret(self, action):
    # expected regret of action (chosen items)
    return np.prod(1 - self.vbar[action]) - np.prod(1 - self.vbar[self.best_action])


''' Run experiments '''
def evaluate_one(Bandit, env, T, period_size, random_seed):
  bandit = Bandit(env.L, T)
  np.random.seed(random_seed)

  regret = np.zeros(T // period_size)
  for t in range(T):
    # generate state
    env.randomize()

    # take action
    action = bandit.get_action(t, env.K)

    # update model and regret
    bandit.update(t, action, env.reward(action))
    regret[t // period_size] += env.regret(action)

  return (regret, bandit)

def evaluate(Bandit, env, num_exps = 5, T = 1000, period_size = 1, display = True):
  if display:
    print("Simulation with %s positions and %s items" % (env.K, env.L))

  seeds = np.random.randint(2 ** 15 - 1, size = num_exps)
  output = [evaluate_one(Bandit, env, T, period_size, seeds[ex]) for ex in range(num_exps)]
  regret = np.vstack([item[0] for item in output]).T
  bandit = output[-1][1]

  if display:
    regretT = np.sum(regret, axis = 0)
    print("Regret: %.2f \\pm %.2f, " % (np.mean(regretT), np.std(regretT) / np.sqrt(num_exps)))

  return (regret, bandit)


def evaluate_one_worker(Bandit, env, T, period_size, seeds,
                        shared_vars, exps):
  """One run of a bandit algorithm."""
  all_regret = shared_vars['all_regret']
  all_alg = shared_vars['all_alg']
  # ex = shared_vars['ex']
  # lock = shared_vars['lock']

  for exp in exps:

    bandit = Bandit(env.L, T)
    np.random.seed(seeds[exp])

    regret = np.zeros(T // period_size)
    for t in range(T):
      # generate state
      env.randomize()

      # take action
      action = bandit.get_action(t, env.K)

      # update model and regret
      bandit.update(t, action, env.reward(action))
      regret[t // period_size] += env.regret(action)

    all_regret[:, exp] = regret
    all_alg[exp] = bandit

    print(".", end="")

def evaluate_parallel(Bandit, env, num_exps=5, T=1000,
                      period_size=1, display=True, num_process=10):
  """Multiple runs of a bandit algorithm in parallel."""
  if display:
    print("Simulation with %s positions and %s items" % (env.K, env.L))
  start = time.time()

  # dots = np.linspace(0, num_exps - 1, 100).astype(int)

  manager = mp.Manager()
  shared_regret = mp.Array('d', np.zeros(T // period_size * num_exps))
  all_regret = np.frombuffer(shared_regret.get_obj()). \
    reshape((T // period_size, num_exps))
  all_alg = manager.list(num_exps * [None])
  exp_dist = np.ceil(np.linspace(0, num_exps, num_process + 1)).astype(int)

  shared_vars = {'all_regret': all_regret, 'all_alg': all_alg}

  seeds = np.random.randint(2 ** 15 - 1, size=num_exps)
  jobs = []
  for i in range(num_process):
    ps = mp.Process(target=evaluate_one_worker,
                    args=(Bandit, env, T, period_size, seeds,
                          shared_vars, range(exp_dist[i], exp_dist[i + 1])))
    jobs.append(ps)
    ps.start()

  for job in jobs:
    job.join()

  if display:
    print(" %.1f seconds" % (time.time() - start))

  if display:
    total_regret = all_regret.sum(axis=0)
    # print(total_regret)
    print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)" %
          (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
           np.median(total_regret), total_regret.max(), total_regret.min()))

  return all_regret, all_alg