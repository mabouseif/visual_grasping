#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    file_path = "/home/mohamed/drive/coppelia_stuff/scripts/multi_metrics.npy"
    all_runs = np.load(file_path)

    completion = []
    grasp_success_rate = []
    action_efficiency = []
    thresh = 7
    n_objects = 6.0

    for i, run in enumerate(all_runs):
        count = 0
        for j, res in enumerate(run):
            count = 0 if res else count+1
            if count == thresh:
                completion.append(False)
                break

        if count != thresh:
            completion.append(True)
            grasp_success_rate.append(np.mean(run))
            action_efficiency.append(n_objects / len(run))


print("Completion: {}".format(np.round(np.mean(completion), 2)))
print("Action Success Rate: {}".format(np.round(np.mean(grasp_success_rate), 2)))
print("Action Efficiency: {}".format(np.round(np.mean(action_efficiency), 2)))