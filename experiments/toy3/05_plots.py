import os
import sys
import pathlib
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Ugly hack
sys.path.insert(0, os.path.abspath(f"."))

from stat_tests import run_test


if __name__ == '__main__':

    # read experiment config
    with open("experiments/toy3/config.json", "r") as json_data_file:
        cfg = json.load(json_data_file)

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'privileged_policy',
        type=str,
        choices=cfg['privileged_policies'].keys(),
    )
    args = parser.parse_args()

    privileged_policy = args.privileged_policy

    print(f"privileged_policy : {privileged_policy }")


    ## COLLECT THE RESULTS ##

    nobss = cfg['nsamples_obs']
    nints = cfg['nsamples_int']
    training_schemes = cfg["training_schemes"]

    nseeds = 20

    model_results = []
    agent_results = []
    for seed in range(nseeds):
        with open(f"experiments/toy3/results/{privileged_policy}/seed_{seed}/model_results.npy", 'rb') as f:
            model_results.append(np.load(f))
        with open(f"experiments/toy3/results/{privileged_policy}/seed_{seed}/agent_results.npy", 'rb') as f:
            agent_results.append(np.load(f))

    model_results = np.asarray(model_results)
    agent_results = np.asarray(agent_results)

    # kls = model_results[..., 0]
    jss = model_results[..., 1]
    # ces = model_results[..., 2]
    rewards = agent_results[..., 0]


    ## CREATE AND SAVE THE PLOTS ##

    plotsdir = pathlib.Path(f"experiments/toy3/plots")
    plotsdir.mkdir(parents=True, exist_ok=True)

    rmin = np.min(rewards)
    rmax = np.max(rewards)

    jsmin = np.min(jss)
    jsmax = np.max(jss)

    r_int = rewards[..., 0]
    r_naiv = rewards[..., 1]
    r_augm = rewards[..., 2]

    js_int = jss[..., 0]
    js_naiv = jss[..., 1]
    js_augm = jss[..., 2]

    fig, axes = plt.subplots(2, 5, figsize=(20, 6), dpi=300)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    ax = axes[0, 0]
    cf = ax.pcolormesh(r_int.mean(0), vmin=rmin, vmax=rmax)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"no obs")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    ax = axes[0, 1]
    cf = ax.pcolormesh(r_naiv.mean(0), vmin=rmin, vmax=rmax)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"naive obs+int")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    ax = axes[0, 2]
    cf = ax.pcolormesh(r_augm.mean(0), vmin=rmin, vmax=rmax)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"augmented obs+int")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    r_gain_int = (r_augm - r_int).mean(0)
    r_gain_naiv = (r_augm - r_naiv).mean(0)
    r_gain_max = np.max([np.abs(r_gain_int), np.abs(r_gain_naiv)])
    r_gain_min = -r_gain_max

    ax = axes[0, 3]
    cf = ax.pcolormesh(r_gain_int, cmap=plt.get_cmap('PiYG'), vmin=r_gain_min, vmax=r_gain_max)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"augmented - no obs")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    ax = axes[0, 4]
    cf = ax.pcolormesh(r_gain_naiv, cmap=plt.get_cmap('PiYG'), vmin=r_gain_min, vmax=r_gain_max)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"augmented - naive")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    ax = axes[1, 0]
    cf = ax.pcolormesh(js_int.mean(0), vmin=jsmin, vmax=jsmax)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"no obs")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    ax = axes[1, 1]
    cf = ax.pcolormesh(js_naiv.mean(0), vmin=jsmin, vmax=jsmax)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"naive obs+int")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    ax = axes[1, 2]
    cf = ax.pcolormesh(js_augm.mean(0), vmin=jsmin, vmax=jsmax)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"augmented obs+int")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    js_gain_int = (js_augm - js_int).mean(0)
    js_gain_naiv = (js_augm - js_naiv).mean(0)
    js_gain_max = np.max([np.abs(js_gain_int), np.abs(js_gain_naiv)])
    js_gain_min = -js_gain_max

    ax = axes[1, 3]
    cf = ax.pcolormesh(js_gain_int, cmap=plt.get_cmap('PiYG'), vmin=js_gain_min, vmax=js_gain_max)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"augmented - no obs")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    ax = axes[1, 4]
    cf = ax.pcolormesh(js_gain_naiv, cmap=plt.get_cmap('PiYG'), vmin=js_gain_min, vmax=js_gain_max)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"augmented - naive")
    ax.set_ylabel('nobs')
    ax.set_xlabel('nints')
    ax.xaxis.set_ticks([i+0.5 for i in range(len(nints))])
    ax.set_xticklabels(nints)
    ax.yaxis.set_ticks([i+0.5 for i in range(len(nobss))])
    ax.set_yticklabels(nobss)

    fig.savefig(plotsdir / f"{privileged_policy}_reward_js_grids.pdf", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    def plot_mean_std(ax, x, y, label, color):
        ax.plot(x, y.mean(0), label=label, color=color)
        ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=color, alpha=0.2)

    def plot_mean_lowhigh(ax, x, mean, low, high, label, color):
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, low, high, color=color, alpha=0.2)

    def compute_central_tendency_and_error(id_central, id_error, sample):
        if id_central == 'mean':
            central = np.nanmean(sample, axis=0)
        elif id_central == 'median':
            central = np.nanmedian(sample, axis=0)
        else:
            raise NotImplementedError

        if isinstance(id_error, int):
            low = np.nanpercentile(sample, q=int((100 - id_error) / 2), axis=0)
            high = np.nanpercentile(sample, q=int(100 - (100 - id_error) / 2), axis=0)
        elif id_error == 'std':
            low = central - np.nanstd(sample, axis=0)
            high = central + np.nanstd(sample, axis=0)
        elif id_error == 'sem':
            low = central - np.nanstd(sample, axis=0) / np.sqrt(sample.shape[0])
            high = central + np.nanstd(sample, axis=0) / np.sqrt(sample.shape[0])
        else:
            raise NotImplementedError

        return central, low, high

    for i, nobs in enumerate(nobss):

        test = 'Wilcoxon'
        deviation = 'std'  # 'sem'
        confidence_level = 0.05

        ### Jensen-Shannon ###

        fig, axes = plt.subplots(1, 1, figsize=(3, 2.25), dpi=300)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # statistical tests
        test_int_augm = [run_test(test, js_augm[:, i, j], js_int[:, i, j], alpha=confidence_level) for j in range(len(nints))]
        test_naiv_augm = [run_test(test, js_augm[:, i, j], js_naiv[:, i, j], alpha=confidence_level) for j in range(len(nints))]

        # mean and standard error
        mean0, low0, high0 = compute_central_tendency_and_error('mean', deviation, js_int[:, i])
        mean1, low1, high1 = compute_central_tendency_and_error('mean', deviation, js_naiv[:, i])
        mean2, low2, high2 = compute_central_tendency_and_error('mean', deviation, js_augm[:, i])

        # plot JS curves
        ax = axes
        plot_mean_lowhigh(ax, nints, mean0, low0, high0, label="no obs", color="tab:blue")
        plot_mean_lowhigh(ax, nints, mean1, low1, high1, label="naive", color="tab:orange")
        plot_mean_lowhigh(ax, nints, mean2, low2, high2, label="augmented", color="tab:green")

        ymax = np.nanmax([high0, high1, high2])
        ymin = np.nanmin([low0, low1, low2])

        # plot significative difference as dots
        y = ymax + 0.05 * (ymax-ymin)
        x = np.asarray(nints)[np.argwhere(test_int_augm)]
        ax.scatter(x, y * np.ones_like(x), s=10, c='tab:blue', marker='v')

        y = ymax + 0.10 * (ymax-ymin)
        x = np.asarray(nints)[np.argwhere(test_naiv_augm)]
        ax.scatter(x, y * np.ones_like(x), s=10, c='tab:orange', marker='s')

        ax.set_title(f"JS divergence")
        ax.set_xlabel('nints (log scale)')
        ax.set_xscale('log', base=2)
        ax.set_ylim(bottom=0)
        ax.legend()

        fig.savefig(plotsdir / f"{privileged_policy}_js_nobs_{nobs}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close(fig)


        ### Reward ###

        fig, axes = plt.subplots(1, 1, figsize=(3, 2.25), dpi=300)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # statistical tests
        test_int_augm = [run_test(test, r_int[:, i, j], r_augm[:, i, j], alpha=confidence_level) for j in range(len(nints))]
        test_naiv_augm = [run_test(test, r_naiv[:, i, j], r_augm[:, i, j], alpha=confidence_level) for j in range(len(nints))]

        # mean and standard error
        mean0, low0, high0 = compute_central_tendency_and_error('mean', deviation, r_int[:, i])
        mean1, low1, high1 = compute_central_tendency_and_error('mean', deviation, r_naiv[:, i])
        mean2, low2, high2 = compute_central_tendency_and_error('mean', deviation, r_augm[:, i])

        # plot reward curves
        ax = axes
        plot_mean_lowhigh(ax, nints, mean0, low0, high0, label="no obs", color="tab:blue")
        plot_mean_lowhigh(ax, nints, mean1, low1, high1, label="naive", color="tab:orange")
        plot_mean_lowhigh(ax, nints, mean2, low2, high2, label="augmented", color="tab:green")

        ymax = np.nanmax([high0, high1, high2])
        ymin = np.nanmin([low0, low1, low2])

        # plot significative difference as dots
        y = ymax + 0.05 * (ymax - ymin)
        x = np.asarray(nints)[np.argwhere(test_int_augm)]
        ax.scatter(x, y * np.ones_like(x), s=10, c='tab:blue', marker='v')

        y = ymax + 0.10 * (ymax - ymin)
        x = np.asarray(nints)[np.argwhere(test_naiv_augm)]
        ax.scatter(x, y * np.ones_like(x), s=10, c='tab:orange', marker='s')

        ax.set_title(f"reward")
        ax.set_xlabel('nints (log scale)')
        ax.set_xscale('log', base=2)
        # ax.legend()

        fig.savefig(plotsdir / f"{privileged_policy}_reward_nobs_{nobs}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
