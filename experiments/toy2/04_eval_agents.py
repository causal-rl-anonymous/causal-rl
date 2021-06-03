import sys
import os
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

    from models import TabularAugmentedModel

    # ENVIRONMENT 
    from environment import PomdpEnv
    from environment.env_wrappers import BeliefStateRepresentation, RewardWrapper, SqueezeEnv

    from rl_agents.ac import ActorCritic, evaluate_agent

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

    reward_map = cfg["r_desc"]


    ## SET UP THE SEEDS ##

    rng = np.random.RandomState(seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_model_training = rng.randint(0, 2**10)
    seed_data_eval = rng.randint(0, 2**10)
    seed_eval = rng.randint(0, 2**10)
    seed_agent_training = rng.randint(0, 2**10)
    seed_agent_eval = rng.randint(0, 2**10)

    ## EVALUATE THE TRANSITION MODELS ##

    nsamples_obs_subsets = cfg['nsamples_obs']
    nsamples_int_subsets = cfg['nsamples_int']
    training_schemes = cfg["training_schemes"]

    ## EVALUATE THE TRANSITION MODELS ##

    n_episodes = 100

    device = torch.device(device)

    # learnt model
    m = TabularAugmentedModel(s_nvals=s_nvals, o_nvals=o_nvals, a_nvals=a_nvals, r_nvals=r_nvals)
    m = m.to(device)

    # learnt agent
    agent = ActorCritic(s_nvals=s_nvals, a_nvals=a_nvals)
    agent.to(device)

    # true POMDP
    env_p = PomdpEnv(p_s=p_s,
                     p_or_s=p_r_s.unsqueeze(-2) * p_o_s.unsqueeze(-1),
                     p_s_sa=p_s_sa,
                     categorical_obs=True,
                     max_length=episode_length)

    resultsdir = pathlib.Path(f"experiments/toy2/results/{privileged_policy}/seed_{seed}")
    resultsdir.mkdir(parents=True, exist_ok=True)

    results = np.full((len(nsamples_obs_subsets), len(nsamples_int_subsets), len(training_schemes), 1), np.nan)

    for i, nsamples_obs in enumerate(nsamples_obs_subsets):
        for j, nsamples_int in enumerate(nsamples_int_subsets):
            for k, training_scheme in enumerate(training_schemes):

                print(f"nsamples_obs: {nsamples_obs} nsamples_int: {nsamples_int} training_scheme: {training_scheme}")

                model_dir = pathlib.Path(f"experiments/toy2/trained_models/{privileged_policy}/seed_{seed}/nobs_{nsamples_obs}/nint_{nsamples_int}")
                model_paramsfile = model_dir / f"{training_scheme}.pt"

                agent_dir = pathlib.Path(f"experiments/toy2/trained_agents/{privileged_policy}/seed_{seed}/nobs_{nsamples_obs}/nint_{nsamples_int}")
                agent_paramsfile = agent_dir / f"{training_scheme}.pt"

                print(f"reading model from: {model_paramsfile}")
                m.load_state_dict(torch.load(model_paramsfile, map_location=device))

                print(f"reading agent from: {agent_paramsfile}")
                agent.load_state_dict(torch.load(agent_paramsfile, map_location=device))

                # POMDP -> MDP (using the model's belief state)
                env = BeliefStateRepresentation(SqueezeEnv(env_p), m)

                # map categorical reward to its numerical values
                env = RewardWrapper(env, reward_dic=reward_map)

                # agent evaluation (true environment)
                torch.manual_seed(seed_agent_eval)

                reward = evaluate_agent(env, agent, n_episodes)

                print(f"reward: {reward}")

                results[i, j, k] = reward

    with open(resultsdir / "agent_results.npy", 'wb') as f:
        np.save(f, results)
