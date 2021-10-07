
##############################################################
##################### WRAPPERS ###############################
##############################################################

# Packages
# import cv2
import gym
import time
import torch

import numpy as np

####################### To MDP ###############################

class SqueezeEnv(gym.Wrapper):
    def __init__(self, env):
        super(SqueezeEnv, self).__init__(env)
        self.env = env

    def reset(self):
        o = self.env.reset()
        return o.unsqueeze(0)       

    def step(self, action):

        o, r, done, info = self.env.step(action)
        return o.unsqueeze(0), r.unsqueeze(0), done.unsqueeze(0), info

class RewardWrapper(gym.Wrapper):

    def __init__(self, env, reward_dic):
        super(RewardWrapper, self).__init__(env)
        self.env = env
        self.reward_dic = reward_dic
        
    def reset(self):
        o = self.env.reset()
        self.r = self.reward_dic[int(self.env.r.argmax())]
        return o    
        
    def step(self, action):

        o, r, done, info = self.env.step(action)
        r = self.reward_dic[int(r.argmax())]
        return o, r, done, info


################## Augmented POMDP #############################

class BeliefStateRepresentation(gym.Wrapper):

    """ Return the same POMDP env with a belief state representation as attribute.
        Compute p(s|h), ie the proba distribution of the hidden state given trajectory history, 
        using the model estimation of p_s_h, and store the computed vector
        as a belief state representation attributes """

    def __init__(self, env, belief_state_model, with_done=False):
        super(BeliefStateRepresentation, self).__init__(env)

        self.env = env
        self.internal_model = belief_state_model
        self.with_done = with_done

        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(belief_state_model.s_nvals,), dtype=np.float)

    def update_belief_state(self, a, o, r, d):

        ''' Update the proba distribution of hidden state representation p(s|h).
        Ie the distribution of the hidden state estimated s given the whole history '''

        with torch.no_grad():
            self.log_q_s_h = self.internal_model.log_q_s_h(
                regime=torch.tensor([1.]),  # interventional regime
                loq_q_sprev_hprev=self.log_q_s_h, 
                a=a, o=o, r=r, d=d, 
                with_done=self.with_done)

    def reset(self):

        ''' Reset hidden state beliefs and last action  '''

        self.log_q_s_h = None  # reset belief state

        o = self.env.reset()
        r = self.env.r.unsqueeze(0)
        d = torch.tensor([0])

        self.update_belief_state(None, o, r, d)

        return torch.exp(self.log_q_s_h) 

    def step(self, action):

        ''' Take a step, update hidden state beliefs and
        return the hidden state p(s|h) as observation '''

        o, r, d, info = self.env.step(action)
        a = torch.tensor([1. if action==i else 0. for i in range(self.action_space.n)]).unsqueeze(0)

        self.update_belief_state(a, o, r, d)

        return torch.exp(self.log_q_s_h), r, d, info

