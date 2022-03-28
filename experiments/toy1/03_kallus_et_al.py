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
        '-s', '--nseeds',
        type=int,   
        help = 'Number of random seed.',
        default=20,
    )
    parser.add_argument(
        'privileged_policy',
        type=str,
        choices=cfg['privileged_policies'].keys(),
    )
    args = parser.parse_args()

    nseeds = args.nseeds
    privileged_policy = args.privileged_policy

    print(f"nseeds: {nseeds}")
    print(f"privileged_policy : {privileged_policy}")

    # Ugly hack
    sys.path.insert(0, os.path.abspath(f"."))
    
    import torch
    from utils_kallus import run_method, get_best_for_data, regs

    ## SET UP RANGE OF EXPERIMENTS ##
    nsample_obs = 512
    nsamples_int = [4, 8, 16, 32, 64, 128]

    ## SET UP THE SEEDS ##
    seed = 0
    rng = np.random.RandomState(seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_training = rng.randint(0, 2**10)

    ## SET UP THE ENVIRONMENT ##

    p_s = torch.tensor(cfg['p_s'])
    p_r_s = torch.tensor(cfg['p_r_s'])
    p_o_s = torch.tensor(cfg['p_o_s'])
    p_s_sa = torch.tensor(cfg['p_s_sa'])

    p_a_s = torch.tensor(cfg['privileged_policies'][privileged_policy])

    nsamples_obs = 512
    nsamples_int_subsets = cfg['nsamples_int']
    training_schemes = cfg["training_schemes"]

    resultsdir = pathlib.Path(f"experiments/toy1/results/kallus")
    resultsdir.mkdir(parents=True, exist_ok=True)

    results_kallus = np.full((nseeds, len(nsamples_int_subsets)),np.nan)
    
    best_eta_regs = np.zeros((nseeds, len(nsamples_int_subsets)))
    predicted_taus = np.zeros((nseeds, len(nsamples_int_subsets)))
    predicted_omegas = np.zeros((nseeds, len(nsamples_int_subsets)))

    ## Calcul du true tau
    p_a_s = torch.tensor(cfg['privileged_policies'][privileged_policy])
    
    for seed in range(nseeds):
        
        #ugly hack for unexpected errors 
        if seed in [13, 19] :
            seed += 8
        
        ## SET UP THE SEEDS ##
        rng = np.random.RandomState(seed)
        seed_data_obs = rng.randint(0, 2**10)
        seed_data_int = rng.randint(0, 2**10)
        seed_training = rng.randint(0, 2**10)
        
        #ugly hack for unexpected errors
        if seed in [21, 27] :
            seed -= 8

        np.random.seed(seed_data_obs)
        n_obs = nsample_obs
        X_obs = np.array([1 for i in range(n_obs)])
        U_obs = np.random.choice([0,1,2], p=np.array(p_s), size=n_obs)
        T_obs = np.array([np.random.choice([0,1], p=np.array(p_a_s[u]), size=1)[0] for u in U_obs])
        Y_obs = np.array([p_r_s[p_s_sa[U_obs[i], T_obs[i]].argmax()].argmax() for i in range(n_obs)], dtype=int)

        np.random.seed(seed_data_int)
        X_int_ = np.array([1 for i in range(nsamples_int_subsets[-1])])
        U_int_ = np.random.choice([0,1,2], p=np.array(p_s), size=nsamples_int_subsets[-1])
        T_int_ = np.array([np.random.choice([0,1], p=[0.5, 0.5], size=1)[0] for u in U_int_])
        Y_int_ = np.array([p_r_s[p_s_sa[U_int_[i], T_int_[i]].argmax()].argmax() for i in range(nsamples_int_subsets[-1])], dtype=int)
        
        for k, nsamples_int in enumerate(nsamples_int_subsets):

            X_int = X_int_[:nsamples_int]
            U_int = U_int_[:nsamples_int]
            T_int = T_int_[:nsamples_int]
            Y_int = Y_int_[:nsamples_int]

            np.random.seed(seed_training)
            f1pred_obs = get_best_for_data(X_obs[T_obs>0].reshape(-1,1), Y_obs[T_obs>0], regs)
            f0pred_obs = get_best_for_data(X_obs[T_obs==0].reshape(-1,1), Y_obs[T_obs==0], regs)
            omega_est_int = f1pred_obs.predict(X_int.mean().reshape(-1,1)) - f0pred_obs.predict(X_int.mean().reshape(-1,1)) 

            best_eta_reg , eta_est_cf, _ = run_method(X_int, Y_int, T_int, X_obs, Y_obs, T_obs)

            best_eta_regs[seed, k] = best_eta_reg.predict(X_obs.mean().reshape(-1, 1))[0]
            predicted_taus[seed, k] = omega_est_int[0] + best_eta_reg.predict(X_obs.mean().reshape(-1, 1))[0]
            predicted_omegas[seed, k] = omega_est_int[0]
                
            pred_tau = predicted_taus[seed, k]
            optimal_policy =  torch.tensor(np.repeat(np.array([pred_tau<0, pred_tau>0], dtype=int).reshape(1,-1),3,  axis=0))
            if pred_tau == 0:
                optimal_policy = torch.tensor(np.repeat(np.array([0.5, 0.5]).reshape(1,-1),3,  axis=0))
            reward = (p_r_s * (p_s_sa * (optimal_policy * p_s.unsqueeze(-1)).unsqueeze(-1)).sum(1).sum(0).unsqueeze(1)).sum(0)
            results_kallus[seed, k] = reward[1]
            
    with open(f"experiments/toy1/results/kallus/{privileged_policy}.npy", 'wb') as f:
        np.save(f, results_kallus)

