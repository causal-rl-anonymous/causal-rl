import os
import sys
import pathlib
import json
import argparse
import numpy as np


if __name__ == '__main__':

    # read experiment config
    with open("experiments/toy1/config.json", "r") as json_data_file:
        cfg = json.load(json_data_file)

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        type=int,
        help = 'Random generator seed.',
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        type=int,
        help='CUDA GPU id (-1 for CPU).',
        default=-1,
    )
    parser.add_argument(
        'privileged_policy',
        type=str,
        choices=cfg['privileged_policies'].keys(),
    )
    args = parser.parse_args()

    # process command-line arguments
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
        device = f"cuda:{args.gpu}"

    seed = args.seed
    privileged_policy = args.privileged_policy

    print(f"device: {device}")
    print(f"seed: {seed}")
    print(f"privileged_policy : {privileged_policy}")


    import torch

    # Ugly hack
    sys.path.insert(0, os.path.abspath(f"."))

    from environment import PomdpEnv
    from policies import UniformPolicy, ExpertPolicy
    from models import TabularAugmentedModel

    from utils import js_div, kl_div


    ## SET UP THE ENVIRONMENT ##

    p_s = torch.tensor(cfg['p_s'])
    p_r_s = torch.tensor(cfg['p_r_s'])
    p_o_s = torch.tensor(cfg['p_o_s'])
    p_s_sa = torch.tensor(cfg['p_s_sa'])

    p_a_s = torch.tensor(cfg['privileged_policies'][privileged_policy])

    o_nvals=p_o_s.shape[1]
    a_nvals=p_s_sa.shape[1]
    r_nvals=p_r_s.shape[1]
    s_nvals = cfg["latent_space_size"]

    # POMDP dynamics
    env = PomdpEnv(p_s=p_s,
                   p_or_s=p_r_s.unsqueeze(-2) * p_o_s.unsqueeze(-1),
                   p_s_sa=p_s_sa,
                   categorical_obs=True,
                   max_length=1)

    # Policy in the observational regime (priviledged)
    obs_policy = ExpertPolicy(p_a_s)

    # Policy in the interventional regime
    int_policy = UniformPolicy(a_nvals)

    # recovering the true bandit transition model
    with torch.no_grad():
        p_ssr_a_int = p_s.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)\
            * p_s_sa.permute(1, 0, 2).unsqueeze(-1) \
            * p_r_s.unsqueeze(0).unsqueeze(0)
        p_r_a_int = p_ssr_a_int.sum(dim=(1, 2))


    ## EVALUATE THE TRANSITION MODELS ##

    nsamples_obs_subsets = cfg['nsamples_obs']
    nsamples_int_subsets = cfg['nsamples_int']
    training_schemes = cfg["training_schemes"]

    with_done = False

    device = torch.device(device)

    resultsdir = pathlib.Path(f"experiments/toy1/results/{privileged_policy}/seed_{seed}")
    resultsdir.mkdir(parents=True, exist_ok=True)

    results = np.full((len(nsamples_obs_subsets), len(nsamples_int_subsets), len(training_schemes), 3), np.nan)

    p_r_a_int = p_r_a_int.to(device)

    m = TabularAugmentedModel(s_nvals=s_nvals, o_nvals=o_nvals, a_nvals=a_nvals, r_nvals=r_nvals)
    m = m.to(device)

    for i, nsamples_obs in enumerate(nsamples_obs_subsets):
         for j, nsamples_int in enumerate(nsamples_int_subsets):
            for k, training_scheme in enumerate(training_schemes):

                print(f"nsamples_obs: {nsamples_obs} nsamples_int: {nsamples_int} training_scheme: {training_scheme}")

                modeldir = pathlib.Path(f"experiments/toy1/trained_models/{privileged_policy}/seed_{seed}/nobs_{nsamples_obs}/nint_{nsamples_int}")

                print(f"reading results from: {modeldir}")

                paramsfile = modeldir / f"{training_scheme}.pt"
                m.load_state_dict(torch.load(paramsfile, map_location=device))

                # recovering the learnt bandit transition model
                with torch.no_grad():
                    q_s = torch.nn.functional.softmax(m.params_s, dim=-1)
                    q_r_s = torch.nn.functional.softmax(m.params_r_s, dim=-1)
                    q_s_sa = torch.nn.functional.softmax(m.params_s_sa, dim=-1)

                    # a, s, s, r
                    q_ssr_a_int = q_s.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)\
                        * q_s_sa.permute(1, 0, 2).unsqueeze(-1) \
                        * q_r_s.unsqueeze(0).unsqueeze(0)
                    q_r_a_int = q_ssr_a_int.sum(dim=(1,2))

                # computing the evaluation measures
                with torch.no_grad():
                    jsd = js_div(p_r_a_int, q_r_a_int).mean(0)  # expectation over uniform policy
                    kld = kl_div(p_r_a_int, q_r_a_int).mean(0)  # expectation over uniform policy
                    reward = p_r_a_int[q_r_a_int[:, 1].argmax(dim=0), 1]

                jsd = jsd.item()
                kld = kld.item()
                reward = reward.item()

                print(f"jsd: {jsd}")
                print(f"kld: {kld}")
                print(f"reward: {reward}")

                results[i, j, k] = (jsd, kld, reward)

    with open(resultsdir / "results.npy", 'wb') as f:
        np.save(f, results)
