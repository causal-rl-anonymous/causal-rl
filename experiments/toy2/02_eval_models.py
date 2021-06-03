import os
import sys
import pathlib
import json
import argparse
import numpy as np


if __name__ == '__main__':

    # read experiment config
    with open("experiments/toy2/config.json", "r") as json_data_file:
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

    from utils import construct_dataset
    from utils import js_div_empirical, kl_div_empirical, cross_entropy_empirical


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

    episode_length = cfg["episode_length"]

    # POMDP dynamics
    env = PomdpEnv(p_s=p_s,
                   p_or_s=p_r_s.unsqueeze(-2) * p_o_s.unsqueeze(-1),
                   p_s_sa=p_s_sa,
                   categorical_obs=True,
                   max_length=episode_length)

    # Policy in the observational regime (priviledged)
    obs_policy = ExpertPolicy(p_a_s)

    # Policy in the interventional regime
    int_policy = UniformPolicy(a_nvals)


    ## SET UP THE SEEDS ##

    rng = np.random.RandomState(seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_training = rng.randint(0, 2**10)
    seed_data_eval = rng.randint(0, 2**10)
    seed_eval = rng.randint(0, 2**10)


    ## GENERATE THE EVALUATION DATASET ##

    nsamples_eval = cfg["nsamples_eval"]

    torch.manual_seed(seed_data_eval)
    data_eval_p = construct_dataset(env=env,
        policy=int_policy,
        n_samples=nsamples_eval,
        regime=torch.tensor(1))


    ## EVALUATE THE TRANSITION MODELS ##

    nsamples_obs_subsets = cfg['nsamples_obs']
    nsamples_int_subsets = cfg['nsamples_int']
    training_schemes = cfg["training_schemes"]

    with_done = False
    batch_size = 32

    device = torch.device(device)

    # true model
    m_true = TabularAugmentedModel(s_nvals=p_s.shape[0], o_nvals=o_nvals, a_nvals=a_nvals, r_nvals=r_nvals)
    m_true.set_probs(p_s=p_s, p_o_s=p_o_s, p_r_s=p_r_s, p_s_sa=p_s_sa, p_a_s=p_a_s)
    m_true.to(device)

    # learnt model
    m = TabularAugmentedModel(s_nvals=s_nvals, o_nvals=o_nvals, a_nvals=a_nvals, r_nvals=r_nvals)
    m = m.to(device)

    resultsdir = pathlib.Path(f"experiments/toy2/results/{privileged_policy}/seed_{seed}")
    resultsdir.mkdir(parents=True, exist_ok=True)

    results = np.full((len(nsamples_obs_subsets), len(nsamples_int_subsets), len(training_schemes), 3), np.nan)

    for i, nsamples_obs in enumerate(nsamples_obs_subsets):
         for j, nsamples_int in enumerate(nsamples_int_subsets):
            for k, training_scheme in enumerate(training_schemes):

                print(f"nsamples_obs: {nsamples_obs} nsamples_int: {nsamples_int} training_scheme: {training_scheme}")

                modeldir = pathlib.Path(f"experiments/toy2/trained_models/{privileged_policy}/seed_{seed}/nobs_{nsamples_obs}/nint_{nsamples_int}")

                print(f"reading results from: {modeldir}")

                paramsfile = modeldir / f"{training_scheme}.pt"
                m.load_state_dict(torch.load(paramsfile, map_location=device))

                # sample from the learnt model
                with torch.no_grad():
                    q_s = torch.nn.functional.softmax(m.params_s, dim=-1)
                    q_r_s = torch.nn.functional.softmax(m.params_r_s, dim=-1)
                    q_o_s = torch.nn.functional.softmax(m.params_o_s, dim=-1)
                    q_s_sa = torch.nn.functional.softmax(m.params_s_sa, dim=-1)

                    # imaginary POMDP dynamics
                    env_q = PomdpEnv(p_s=q_s,
                                     p_or_s=q_r_s.unsqueeze(-2) * q_o_s.unsqueeze(-1),
                                     p_s_sa=q_s_sa,
                                     categorical_obs=True,
                                     max_length=episode_length)

                    # imaginary data
                    torch.manual_seed(seed_eval)
                    data_eval_q = construct_dataset(env=env_q,
                                                    policy=int_policy,
                                                    n_samples=nsamples_eval,
                                                    regime=torch.tensor(1))

                    # compute empirical cross-entropy (NLL)
                    ce = cross_entropy_empirical(model_q=m, data_p=data_eval_p,
                                                 batch_size=batch_size, with_done=with_done)

                    # compute empirical KL
                    kld = kl_div_empirical(model_q=m, model_p=m_true,
                                           data_p=data_eval_p,
                                           batch_size=batch_size, with_done=with_done)

                    # compute empirical JS
                    jsd = js_div_empirical(model_q=m, model_p=m_true,
                                           data_q=data_eval_q, data_p=data_eval_p,
                                           batch_size=batch_size, with_done=with_done)

                ce = ce.item()
                kld = kld.item()
                jsd = jsd.item()

                print(f"ce: {ce}")
                print(f"kld: {kld}")
                print(f"jsd: {jsd}")

                results[i, j, k] = (kld, jsd, ce)

    with open(resultsdir / "model_results.npy", 'wb') as f:
        np.save(f, results)
