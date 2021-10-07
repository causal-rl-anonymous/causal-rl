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
        '--nobs',
        type=int,
        help = 'Number of observational samples.',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nint',
        type=int,
        help = 'Number of interventional samples.',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--scheme',
        type=str,
        choices=cfg['training_schemes'],
        help='Training scheme.',
        default=argparse.SUPPRESS,
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

    from rl_agents.ac import ActorCritic, run_actorcritic
    # from rl_agents.reinforce import Actor, run_reinforce

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

    ## EVALUATE THE TRANSITION MODELS ##

    # from command-line argument if provided, otherwise from config file
    nsamples_obs_subsets = [args.nobs] if "nobs" in args else cfg['nsamples_obs']
    nsamples_int_subsets = [args.nint] if "nint" in args else cfg['nsamples_int']
    training_schemes = [args.scheme] if "scheme" in args else cfg["training_schemes"]

    print(f"nsamples_obs_subsets: {nsamples_obs_subsets}")
    print(f"nsamples_int_subsets: {nsamples_int_subsets}")
    print(f"training_schemes: {training_schemes}")

    ## EVALUATE THE TRANSITION MODELS ##

    with_done = False
    lr = 1e-2
    gamma = 0.9
    n_epochs = 1000
    log_every = 10
    batch_size = 8

    device = torch.device(device)

    # learnt model
    m = TabularAugmentedModel(s_nvals=s_nvals, o_nvals=o_nvals, a_nvals=a_nvals, r_nvals=r_nvals)
    m = m.to(device)

    for nsamples_obs in nsamples_obs_subsets:
        for nsamples_int in nsamples_int_subsets:
            for training_scheme in training_schemes:

                print(f"nsamples_obs: {nsamples_obs} nsamples_int: {nsamples_int} training_scheme: {training_scheme}")

                model_dir = pathlib.Path(f"experiments/toy2/trained_models/{privileged_policy}/seed_{seed}/nobs_{nsamples_obs}/nint_{nsamples_int}")
                agent_dir = pathlib.Path(f"experiments/toy2/trained_agents/{privileged_policy}/seed_{seed}/nobs_{nsamples_obs}/nint_{nsamples_int}")

                agent_dir.mkdir(parents=True, exist_ok=True)

                model_paramsfile = model_dir / f"{training_scheme}.pt"
                agent_paramsfile = agent_dir / f"{training_scheme}.pt"
                logfile = agent_dir / f"{training_scheme}_log.txt"

                if agent_paramsfile.exists():
                    print(f"Found trained agent {agent_paramsfile}, skip training.")
                    continue

                print(f"reading model from: {model_paramsfile}")

                m.load_state_dict(torch.load(model_paramsfile, map_location=device))

                # recover learned POMDP dynamics
                with torch.no_grad():
                    q_s = torch.nn.functional.softmax(m.params_s, dim=-1)
                    q_r_s = torch.nn.functional.softmax(m.params_r_s, dim=-1)
                    q_o_s = torch.nn.functional.softmax(m.params_o_s, dim=-1)
                    q_s_sa = torch.nn.functional.softmax(m.params_s_sa, dim=-1)

                # learned POMDP
                env_q = PomdpEnv(p_s=q_s,
                                 p_or_s=q_r_s.unsqueeze(-2) * q_o_s.unsqueeze(-1),
                                 p_s_sa=q_s_sa,
                                 categorical_obs=True,
                                 max_length=episode_length)

                # POMDP -> MDP (using the model's belief state)
                env_q = BeliefStateRepresentation(SqueezeEnv(env_q), m)

                # map categorical reward to numerical values
                env_q = RewardWrapper(env_q, reward_dic=reward_map)

                # agent training (dream)
                torch.manual_seed(seed_agent_training)

                # agent = Actor(s_nvals=s_nvals, a_nvals=a_nvals)
                # run_reinforce(env=env_q, agent=agent,
                #               lr=lr, gamma=gamma,
                #               batch_size=batch_size,
                #               n_epochs=n_epochs,
                #               log_every=log_every,
                #               logfile=logfile)

                agent = ActorCritic(s_nvals=s_nvals, a_nvals=a_nvals)
                run_actorcritic(env_q, agent,
                                lr=lr, gamma=gamma,
                                batch_size=batch_size,
                                n_epochs=n_epochs,
                                log_every=log_every,
                                logfile=logfile)

                torch.save(agent.state_dict(), agent_paramsfile)
                print(f"saving agent to: {agent_paramsfile}")
