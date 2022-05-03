from tokenize import Double
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


class Bandit(object):
    def __init__(self, p_real):
        self.p_real = p_real
        self.p_estimate = 0.0
        self.N = 0

    def pull(self) -> int:
        return 0 if np.random.rand() >= self.p_real else 1

    def update(self, x: int) -> None:
        self.N += 1
        self.p_estimate = (x + self.p_estimate * (self.N - 1)) / self.N
        

class ExperimentEpsilonGreedy(object):
    def __init__(self, number_of_tries: int, p_real_list: list[float], epsilon=0.1):
        self.number_of_tries = number_of_tries
        self.bandits = [Bandit(p) for p in p_real_list]
        self.epsilon = epsilon

    def get_max_idx(self, v):
        arr = np.array(v)
        idx = np.where(arr == arr.max())[0]
        return np.random.choice(idx)
    
    def ucb(self, mean, n, nj):
        return mean + np.sqrt(np.log(n) / nj)

    def best_possible_outcome(self) -> float:
        best_bandit_index = np.argmax([b.p_real for b in self.bandits])
        return self.bandits[best_bandit_index].p_real * (1 - self.epsilon) + np.mean([b.p_real for b in self.bandits]) * self.epsilon

    @timing
    def run(self):
        best_bandit_count = 0
        best_bandit_index = np.argmax([b.p_real for b in self.bandits])
        bandits_number = len(self.bandits)
        outcomes = np.zeros(self.number_of_tries)
        best_bandit_selected = np.zeros(self.number_of_tries)
        
          # initialization: play each bandit once
        for b in self.bandits:
            x = b.pull()
            b.update(x)

        for i in range(self.number_of_tries):
            current_bandit_index = np.argmax([self.ucb(b.p_estimate, bandits_number, b.N) for b in self.bandits])
            # print(f"current_bandit_index: {current_bandit_index}, {self.bandits[current_bandit_index].N}")

            outcome = self.bandits[current_bandit_index].pull()
            outcomes[i] = outcome
            self.bandits[current_bandit_index].update(outcome)
            if current_bandit_index == best_bandit_index:
                best_bandit_count += 1
                best_bandit_selected[i] = 1

        print(f"-----------------------------------------------------------------------")
        print(f"epsilon: {self.epsilon}")
        print(f"Number of times best score was chosen: {np.sum(best_bandit_selected)}")
        best_possible_score = self.bandits[best_bandit_index].p_real
        print(f"The best bandit real value was {best_possible_score}")
        final_results = outcomes.sum() / outcomes.size
        print(f"The outcome ratio was {final_results}")

        print(f"Using the {self.epsilon} epsilon, the best outcome would be {self.best_possible_outcome()}")
        print(f"-----------------------------------------------------------------------")
        plt.plot(np.cumsum(outcomes) / np.cumsum(np.ones(self.number_of_tries)), label="outcomes")
        plt.plot(np.cumsum(best_bandit_selected) / np.cumsum(np.ones(self.number_of_tries)), label="best_bandit_selected")
        plt.plot(np.ones(self.number_of_tries) * self.best_possible_outcome(), label="best_possible_outcome")
        plt.plot(np.ones(self.number_of_tries) * best_possible_score, label="best_possible_score")
        plt.title(f"epsilon: {self.epsilon}")
        plt.legend(loc="lower right")
        plt.show()


e = ExperimentEpsilonGreedy(10000, [0.2, 0.5, 0.75])
e.run()
