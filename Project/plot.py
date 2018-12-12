import pickle
from os import listdir
import os
from os.path import isfile, join
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


sns.set()
envs = ['CartPole.png', 'MountainCar.png']
dirs_CP = [d[0] for d in os.walk('./') if d[0].startswith('./CP')]
dirs_MC = [d[0] for d in os.walk('./') if d[0].startswith('./MC')]

axis_ranges = [[-5, 205],[-0.55,0.75]]

smooth_values= [10,100]

for env, directories in enumerate([dirs_CP, dirs_MC]):
    f, axes = plt.subplots(2, 2, figsize=(10, 10))
    # f.set_figheight(5)
    # f.set_figwidth(5)

    exp_replay = {'off': axes[0, 0], 'prioritized': axes[0,1], 'uniform': axes[1,env]}
    coloring = {'off': 'r', 'soft': 'b', 'hard': 'g'}

    lines = {}
    errors_upper = {}
    errors_lower = {}
    for directory in directories:
        files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.startswith('steps_run')]
        if directory.startswith('./MC'):
            files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.startswith('maxdistance')]
        runs = []
        for file in files:
            runs.append(pickle.load(open(directory + '/' + file, "rb")))
        stderr = [np.std(x) for x in zip(*runs)]
        a = [x[0] for x in runs]
        runs = [sum(x) for x in zip(*runs)]
        runs = [x / 25 for x in runs]
        runs_lower = np.array(smooth([x - y for x, y in zip(runs, stderr)], smooth_values[env]))
        runs_upper = np.array(smooth([x + y for x, y in zip(runs, stderr)], smooth_values[env]))
        runs = np.array(smooth(runs, smooth_values[env]))
        lines[directory] = runs
        errors_upper[directory] = runs_upper
        errors_lower[directory] = runs_lower

    for x in lines.keys():
        for item in exp_replay.keys():
            if x.endswith(item):
                result = re.search('_(.*)_', x).group(1)
                exp_replay[item].plot(lines[x], label=result, color=coloring[result])
                exp_replay[item].set_title(item)
                # print(runs_lower)
                exp_replay[item].fill_between(range(len(lines[x])), errors_lower[x], errors_upper[x], alpha=0.15)

                exp_replay[item].legend(loc=2)
                exp_replay[item].set_ylim(axis_ranges[env][0], axis_ranges[env][1])

                exp_replay[item].legend()
                handles, labels = exp_replay[item].get_legend_handles_labels()
                # sort both labels and handles by labels

                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                exp_replay[item].legend(handles, labels, loc=2)
    axes[1, 1 - env].set_visible(False)

    if not os.path.exists('./images'):
        os.makedirs('./images')
    plt.savefig('./images/' + envs[env], dpi=300)




