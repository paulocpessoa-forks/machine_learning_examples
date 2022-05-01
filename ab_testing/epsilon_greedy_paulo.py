from tokenize import Double
import numpy as np

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print ('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

class Bandit(object):
    def __init__(self, p_real):
        self.p_real = p_real
        self.p_estimate = 0
        self.N = 0

    def pull(self) -> int:
        return 0 if np.random.rand() >= self.p_real else 1

    def update(self, x: int):
        self.N += 1
        self.p_estimate = (x + self.p_estimate * (self.N - 1)) / self.N


class ExperimentEpsilonGreedy(object):
    def __init__(self, pulls_number: int, p_real_list: list[float], epsilon=0.1):
        self.pulls_number = pulls_number
        self.bandits = [Bandit(p) for p in p_real_list]
        self.epsilon = epsilon

    def get_max_idx(self, v):
        arr = np.array(v)
        idx = np.where(arr == arr.max())[0]
        return np.random.choice(idx)

    @timing
    def run(self):
        best_bandit_count = 0
        best_bandit_index = np.max([b.p_estimate for b in self.bandits])
        bandits_number = len(self.bandits)
        outcomes = np.zeros(self.pulls_number)

        for i in range(self.pulls_number):
            current_bandit_index = 0
            if np.random.rand() > self.epsilon:
                current_bandit_index = self.get_max_idx(
                    [b.p_estimate for b in self.bandits]
                )

            else:
                current_bandit_index = np.random.randint(bandits_number)

            outcome = self.bandits[current_bandit_index].pull()
            outcomes[i] = outcome
            self.bandits[current_bandit_index].update(outcome)
            if current_bandit_index == best_bandit_index:
                best_bandit_count += 1

        print(f"-----------------------------------------------------------------------")
        print(f"The best bandit real value was {self.bandits[best_bandit_index].p_real}")
        final_results = outcomes.sum() / outcomes.size        
        print(f"The outcome ratio was {final_results}")
        
        best_possible_outcome = (
            self.bandits[best_bandit_index].p_real * (1 - self.epsilon)
            + np.mean([b.p_real for b in self.bandits]) * self.epsilon
        )        
        print(
            f"Using the {self.epsilon} epsilon, the best outcome would be {best_possible_outcome}"
        )
        print(f"-----------------------------------------------------------------------")
        


# b = Bandit(0.3)
# b.update(b.pull())
# b.update(b.pull())
# b.update(b.pull())
# b.update(b.pull())
# print(b.p_estimate)
e = ExperimentEpsilonGreedy(10000, [0.2, 0.5, 0.75])
e.run()


# b = Bandit(0.4)
# [b.update(b.pull()) for _ in range(10000)]
# print(b.p_estimate)
