################################################
#################### GYM POMDP #################
################################################

# Packages
import torch
import numpy as np

# Gym
import gym
from gym import spaces

class PomdpEnv(gym.Env):

    def __init__(self, p_s, p_or_s, p_s_sa, max_length, categorical_obs=False):

        """ Environment for POMDP. Needs :
            - p(s) initial distribution for latent variable.
            - p(o|s) distribution of noisy observation given latent state.
            - p(s|s,a) transition distribution of next latent state given current action and state.
            - the maximum length of a rollout.
        """

        self.episode_length = max_length
        self.categorical_obs = categorical_obs

        # Distribution
        self.p_s_sa = p_s_sa
        self.p_s = p_s
        self.p_or_s = p_or_s

        # Initialize game indicators
        self.initialize_on_reset()
        self.action = 0
        self.n_rewards = p_or_s.shape[2]

        # Action and Obersvation Space
        self.action_space = spaces.Discrete(p_s_sa.shape[1])  
        if categorical_obs :
            self.observation_space = spaces.Box(low=0, high=1, shape=(p_or_s.shape[1],), dtype=np.uint8)

        else :
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8)

    def initialize_on_reset(self):

        """ Reset the state of the environment to an initial state. """
        
        self.current_step = 0 
        self.score = 0 
        self.done = False
    
    def reset(self):

        """ Reset Environement : 
            1. Draw initial hidden state from (s).
            2. Get observation from joint p(o, r|s).
            3. Return observation. 
        """
    
        # Reset the state of the environment to an initial state
        self.initialize_on_reset()

        # Draw Initial latent state S
        if self.categorical_obs :
            self.s =  torch.distributions.one_hot_categorical.OneHotCategorical(probs=self.p_s,).sample()
        else :
            self.s = torch.multinomial(self.p_s, 1)

        # Sample o, r from p(o,r|s)
        self.o, self.r = self.sample_ro_s()
        #self.r = torch.tensor(int(r), dtype=torch.float)

        # Return only observation (gym like)
        return self.o


    def sample_s_sa(self):

        """ Sample next hidden state from current state and action p(s|s,a) """

        if self.categorical_obs:
            s = self.s.argmax()
        else :
            s = self.s[0]

        probs = self.p_s_sa[s,self.action,:]
        if self.categorical_obs :
            return torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs,).sample()
        return torch.multinomial(probs, 1)

    def sample_ro_s(self):

        """ Sample reward and observation from joint (conditional) distribution p(o, r|s)"""

        # If categorical var, find corresponding value to index tables
        if self.categorical_obs:
            s = self.s.argmax()

        else :
            s = self.s.clone()

        # Sample from joint multinomial. 
        ind = torch.multinomial(self.p_or_s[s.reshape(-1),:].reshape(s.reshape(-1).size(0), -1), 1)
        size = torch.tensor(self.p_or_s[0].size(), dtype=torch.float)
        ro = torch.cat([ind//size[1], ind%size[1]], dim = -1).reshape(-1)

        if self.categorical_obs :
            o = torch.zeros(self.p_or_s.shape[1])
            o[int(ro[0])] = 1.

            r = torch.zeros(self.p_or_s.shape[2])
            r[int(ro[1])] = 1.
        else :
            o = ro[:1]
            r = ro[1:]

        return o, r


    def step(self, action):

        """ Take a step in the env :
            1. Generate new hidden state p(s|s,a)
            2. Get Obeservation, Reward, Flag Done from p(o,r|s)
            3. return obs, reward, flag_done, info (s) 
        """

        # Increase current steptime
        self.current_step += 1
        # Save picked action
        self.action = action
        # Generate new latent state
        self.s = self.sample_s_sa()

        # Sample obs and reward from hidden state
        self.o, self.r = self.sample_ro_s()

        observation = self.o
        done = int(self.current_step >= self.episode_length)
        reward = self.r

        #Increment total score
        self.score += reward
        return observation, reward, torch.tensor(done, dtype=torch.float), {"s" : self.s}
        
