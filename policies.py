##########################################################
######################## Policies ########################
##########################################################

"""
One can use two kinds of policies to do rollout of the game :

    - UniformPolicy() which plays randomly from the tuple (o, r, d)
    - ExpertPolicy(p_a_s) which mimics an expert that has access to the true state `s`.
     One needs to provide the action distribution `p_a_s`.

In both cases, one would only need to call of the `.action()` property of the Policy class, 
providing all the environment data. As a remark, it returns a one-hot encoded action, 
and the gym environment requires to have an *int*-type action (just use `action.argmax()` then).

    action = policy.action(o, r, done, **info)

One can collect episodes by simply calling `rollout`, given an env and a policy: 
    
    episode = rollout(env, default_policy)

"""

import torch

class Policy(torch.nn.Module):

    """ Empty class, must include
        1. a Reset method
        3. a Action method to act in environment from 4-uplets (o, r, d, info)
    """ 

    def reset(self):
        raise NotImplemented

    def action(self, o, r, d, **info):
        raise NotImplemented
        
        
class UniformPolicy(Policy):

    """ Uniform policy to act within the environement with random distributed actions """

    def __init__(self, a_nvals):
        super().__init__()

        self.a_nvals = a_nvals
        self.h = []

    def reset(self):
        self.h.clear()

    def action(self, o, r, d, **info):

        self.h += [o, r, d]
        a = torch.distributions.one_hot_categorical.OneHotCategorical(\
            probs=torch.ones(self.a_nvals)/self.a_nvals).sample(o.shape[:-1])
        self.h += [a]
 
        return a

class ExpertPolicy(Policy):

    """ Expert Policy Class that chooses its actions from the hidden state s """

    def __init__(self, p_a_s):
        super().__init__()
        self.probs_a_s = p_a_s #p(a_t | s_t, i=0)

    def reset(self):
        pass

    def action(self,  o, r, done, **info):

        s_index = info["s"].argmax()
        a = torch.distributions.one_hot_categorical.OneHotCategorical(probs=self.probs_a_s[s_index], ).sample()

        return a 


class AugmentedPolicy(Policy):

    """ Augmented Policy which uses am augmented model to estimate the belived state
    of the environment. It then chooses its actions according to the believed state estimate."""

    def __init__(self, augmentedmodel, regime=torch.tensor(0), with_done=False):
        super().__init__()

        self.m = augmentedmodel
        self.q_s_h = None
        self.last_action = None
        self.regime = regime
        
    def update_hidden_state(self, o, r, d, with_done=False):
        log_q_s_h = self.m.log_q_s_h(regime=self.regime,
                                     loq_q_sprev_hprev=self.hidden_state, 
                                     a=self.last_action,
                                     o=o.unsqueeze(0), 
                                     r=r.unsqueeze(0),
                                     d=d.unsqueeze(0), 
                                     with_done=False)
        self.q_s_h = torch.exp(log_q_s_h)

    def reset(self):
        self.q_s_h = None
        self.last_action = None
    
    def action(self,  o, r, done, deterministic=False, **info):
        
        print(self.hidden_state, self.last_action)
        self.update_hidden_state(o, r, done)
        new_action_p = torch.exp(self.m.log_q_a_s(a=None, s=self.q_s_h))

        if deterministic :
            new_action_p = new_action_p.round() 

        a = torch.distributions.one_hot_categorical.OneHotCategorical(
                probs=new_action_p,).sample()
        
        self.last_action = a

        return a 

