from __future__ import print_function, division
from ast import Num
from builtins import range
from typing import List

import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class BanditArm:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0.0
        self.N = 0.0

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.0
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

def choose_arg_max(a):
    idx = np.argwhere(np.max(a) == a).flatten()
    return np.random.choice(idx)

def experiment():
    bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_bandit_index = np.argmax([b.p] for b in bandits)
    print("optimal_bandit_index:", optimal_bandit_index)

    for i in range(NUM_TRIALS):
        #use epsilon greedy to select the next bandit
        if np.random.random() < EPS:
            num_times_explored += 1
            bandit_chosen = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            bandit_chosen = choose_arg_max([b.p_estimate for b in bandits])
        
        if bandit_chosen == optimal_bandit_index:
            num_optimal += 1

        reward = bandits[bandit_chosen].pull()

        rewards[i] = reward

        bandits[bandit_chosen].update(reward)

    # print mean estimates for each bandit
    for b in bandits:
        print("mean estimate:", b.p_estimate)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == "__main__":
    experiment()