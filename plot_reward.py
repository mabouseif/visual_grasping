#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# file_path = "/home/mohamed/drive/coppelia_stuff/scripts/logs/2020-05-21.03:30:52/transitions/reward-value.log.txt" # multi
file_path = "/home/mohamed/drive/coppelia_stuff/scripts/logs/mico_cup_1/transitions/reward-value.log.txt"
f = open(file_path)

rewards = []
with open(file_path, 'r') as fp:
    line = fp.readline()
    cnt = 1
    while line:
        # print("Line {}: {}".format(cnt, line.strip()))
        rewards.append(float(line.strip()))
        line = fp.readline()
        cnt += 1

rewards = rewards[1:]
# rewards = rewards[:2500]

grasps = np.asarray(rewards, dtype=np.int32)
avg_rewards = []
avg_grasps = []
window = 100 # 200 for multi
for i, r in enumerate(rewards):
    start_idx = max(0, i-window)
    end_idx = i
    den = window# min(end_idx-start_idx, window)
    avg_rewards.append((i/window)*np.sum(rewards[start_idx:end_idx]) / float(den))
    # avg_grasps.append((i/window)*np.sum(grasps[start_idx:end_idx]) / float(den))
    # avg_grasps.append(np.sum(grasps[start_idx:end_idx]) *0.01)
    avg_grasps.append(np.mean(grasps[start_idx:end_idx])) # *0.01) # for multi

# plt.plot(np.linspace(0, len(rewards)-window, len(rewards)-window), avg_rewards[:-window])
# plt.ylabel("Average Reward")

plt.plot(np.linspace(0, len(rewards)-window, len(rewards)-window), avg_grasps[:-window])
plt.ylabel("Average Grasp Success")

plt.xlabel("Iterations")
# plt.legend(["Cup Grasp"], loc=4)
# plt.savefig('/home/mohamed/Downloads/thesis_stuff/multi_avg_reward.png')
plt.show()
