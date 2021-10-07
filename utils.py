import torch
import datetime


def print_log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


#################################### Metrics ####################################

@torch.jit.script
def kl_div(p, q, ndims: int=1):
#     div = torch.nn.functional.kl_div(p, q, reduction='none')
    div = p * (torch.log(p) - torch.log(q))
    div[p == 0] = 0  # NaNs quick fix
    dims = [i for i in range(-1, -(ndims+1), -1)]
    div = div.sum(dims)
    return div

@torch.jit.script
def js_div(p, q, ndims: int=1):
    m = (p + q) / 2
    div = (kl_div(p, m, ndims) + kl_div(q, m, ndims)) / 2
    return div

#############################################################################

def rollout(env, policy):

    """ Perform rollout of the game and returning episodes in a list """

    episode = []
    o = env.reset()
    r, done, info = env.r, torch.tensor(0.), {"s" : env.s}
    episode += [o, r, done]

    while not done :
        action = policy.action(o, r, done, **info)
        o, r, done, info = env.step(action.argmax())
        episode += [action, o, r, done]
    return episode

def construct_dataset(env, policy, n_samples, regime):

    """ Construct a dataset (of n samples) by collecting rollouts using a given 
        policy in a given environment """

    data = []
    for _ in range(n_samples):
        policy.reset()
        episode = rollout(env, policy)
        data.append((regime, episode))
    return data

#################################################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

####################################### Empirical JS #######################################

from environment.env_pomdp import PomdpEnv

def empiricalJS(model_q, model_p, policy, max_length=1, n_iter=500):
    
    settings_mp = model_p.get_settings()
    env_p = PomdpEnv(p_s=settings_mp["p_s"], p_or_s=settings_mp["p_or_s"], p_s_sa=settings_mp["p_s_sa"],
                     categorical_obs = True, max_length=max_length)  
    
    settings_mq = model_q.get_settings()
    env_q = PomdpEnv(p_s=settings_mq["p_s"], p_or_s=settings_mq["p_or_s"], p_s_sa=settings_mq["p_s_sa"],
                     categorical_obs = True, max_length=max_length)  

    
    n_iter = n_iter
    loss_q, loss_p = 0, 0
    
    # E x~q(x) [log(q(x)) - log(q(x) + p(x))]
    for _ in range(n_iter):
        ep = rollout(env_q, policy)
        ep = [t.unsqueeze(0) for t in ep]
        regime = torch.tensor(1.).unsqueeze(0)
        
        loss_q += model_q.log_prob(regime, ep)[0]
        loss_q -= torch.logsumexp(torch.cat([model_q.log_prob(regime, ep), \
                                             model_p.log_prob(regime, ep)]).unsqueeze(0), 1)[0]


    # E x~p(x) [log(p(x)) - log(q(x) + p(x))]
    for _ in range(n_iter):
        ep = rollout(env_p, policy)
        ep = [t.unsqueeze(0) for t in ep]
        regime = torch.tensor(1.).unsqueeze(0)
        
        loss_p += model_p.log_prob(regime, ep)[0]
        loss_p -= torch.logsumexp(torch.cat([model_q.log_prob(regime, ep), \
                                             model_p.log_prob(regime, ep)]).unsqueeze(0), 1)[0]
        
    return torch.log(torch.tensor(2.)) + (loss_p + loss_q)/(2*n_iter)


def cross_entropy_empirical(model_q, data_p, batch_size, with_done=False):

    device = next(model_q.parameters()).device

    dataloader_p = torch.utils.data.DataLoader(Dataset(data_p), batch_size=batch_size)

    ce = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)

        ce += -log_prob_q.sum(dim=0)

    ce /= len(data_p)

    return ce


def kl_div_empirical(model_p, model_q, data_p, batch_size, with_done=False):

    assert next(model_q.parameters()).device == next(model_p.parameters()).device

    device = next(model_p.parameters()).device

    # Build DataLoaders
    dataloader_p = torch.utils.data.DataLoader(Dataset(data_p), batch_size=batch_size)

    # KL(p|q) = E x~p(x) [log(p(x)) - log(q(x))]
    kl_p_q = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)

        kl_p_q += (log_prob_p - log_prob_q).sum(dim=0)

    kl_p_q /= len(data_p)

    return kl_p_q


def js_div_empirical(model_q, model_p, data_q, data_p, batch_size, with_done=False):

    assert next(model_q.parameters()).device == next(model_p.parameters()).device

    device = next(model_p.parameters()).device

    # Build DataLoaders
    dataloader_q = torch.utils.data.DataLoader(Dataset(data_q), batch_size=batch_size)
    dataloader_p = torch.utils.data.DataLoader(Dataset(data_p), batch_size=batch_size)

    # m = (p + q) / 2

    # KL(p|m) = E x~p(x) [log(p(x)) - log(q(x) + p(x)) + log(2)]
    kl_p_m = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)
        log_prob_m = torch.logsumexp(torch.stack([log_prob_q, log_prob_p], dim=0), dim=0)  # - torch.log(torch.tensor(2, device=device))

        kl_p_m += (log_prob_p - log_prob_m).sum(dim=0)

    kl_p_m /= len(data_p)
    kl_p_m += torch.log(torch.tensor(2, device=device))

    # KL(q|m) = E x~q(x) [log(q(x)) - log(q(x) + p(x)) + log(2)]
    kl_q_m = 0

    for batch in dataloader_q:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)
        log_prob_m = torch.logsumexp(torch.stack([log_prob_q, log_prob_p], dim=0), dim=0)  # - torch.log(torch.tensor(2, device=device))

        kl_q_m += (log_prob_q - log_prob_m).sum(dim=0)

    kl_q_m /= len(data_q)
    kl_q_m += torch.log(torch.tensor(2, device=device))

    # JS(p|q) = (KL(p|m) + KL(q|m)) / 2

    return (kl_q_m + kl_p_m) / 2


from collections import Counter
def get_sampler_weights(data):
    # Get ratio Interventional/Observation
    indices_count = Counter([int(source) for source, ep in data])
    
    # If there is more observational data that interventional, re-weigth the train data sampling
    if indices_count[0] > indices_count[1] : 
    #if indices_count[0] > indices_count[1] : 
        # 1/(2*Nint) for interventional data, 1/(2*Nobs) for obsevational data
        weights = [1./(2*indices_count[int(source)]) for source, ep in data]
        #weigths = [3./(4*indices_count[int(source)]) if source == torch.tensor(0) else 1./(4*indices_count[int(source)]) for source, ep in train_data]
        return weights
    else : 
        return [1 for source, ep in data]

