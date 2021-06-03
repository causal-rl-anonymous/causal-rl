import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

eps = np.finfo(np.float32).eps.item()

############################################################

class ActorCritic(nn.Module):
    
    def __init__(self, s_nvals, a_nvals, hidden_size=32):

        super(ActorCritic, self).__init__()

        self.n_actions = a_nvals
        self.input_shape = s_nvals
        self.hidden_size = hidden_size

        self.dense1 = nn.Linear(self.input_shape, self.hidden_size)
        # self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.critic = nn.Linear(self.hidden_size, 1)
        self.actor = nn.Linear(self.hidden_size, self.n_actions)

    def forward(self, state):
        x = state
        x = nn.ReLU()(self.dense1(x))
        # x = nn.ReLU()(self.dense2(x))
        return nn.LogSoftmax(dim=1)(self.actor(x)), self.critic(x)
    
def run_episode(env, model, max_steps_per_episode):

    action_log_probs, rewards, values = [], [], []

    with torch.no_grad():
        state = env.reset()

    for t in range(max_steps_per_episode):

        action_log_probs_t, value = model.forward(state)
        action = int(torch.multinomial(action_log_probs_t.exp(), 1)[0])

        with torch.no_grad():
            state, reward, done, info = env.step(action)

        action_log_probs.append(action_log_probs_t[:,action])
        values.append(value)
        rewards.append(reward)

        if done: 
            break

    return action_log_probs, values, rewards

def get_expected_return(rewards, gamma, standardize=True):
    
    n = len(rewards)
    discounted_sum = 0.
    returns = np.zeros(n)
    
    for i in range(0, n)[::-1]:
        reward = rewards[i]
        discounted_sum = reward + gamma*discounted_sum
        returns[i] = discounted_sum

    if standardize: 
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)

    return np.array([returns])

# The actor-critic loss
def compute_loss(action_log_probs, values, returns):
    
    actor_loss = - torch.sum(action_log_probs*(returns - values), dim=1)
    critic_loss =  F.smooth_l1_loss(values, returns, reduction = 'sum')

    return actor_loss + critic_loss

def loss_episode(env, model, gamma, max_steps_per_episode):

    action_log_probs, values, rewards = run_episode(env, model, max_steps_per_episode) 

    action_log_probs = torch.cat(action_log_probs,dim=-1)
    values = torch.cat(values,dim=-1)

    returns = get_expected_return(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float, requires_grad=False)

    # Calculating loss values to update our network
    loss = compute_loss(action_log_probs, values, returns)

    return loss, np.sum(rewards)

def run_actorcritic(env, agent,
                    gamma=0.99,
                    n_epochs=10000,
                    batch_size=1,
                    max_steps_per_episode=1000,
                    log_every=1000,
                    lr=1e-2):

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    best_running_return = -float("inf")
    best_params = agent.state_dict().copy()

    for ep in range(n_epochs) :

        batch_return = 0
        optimizer.zero_grad()

        for i in range(batch_size):
            loss, episode_return = loss_episode(env, agent, gamma, max_steps_per_episode)
            batch_return += episode_return / batch_size
            loss = loss / batch_size
            loss.backward()

        if ep == 0:
            running_return = batch_return
        else:
            running_return = batch_return * 0.1 + running_return * 0.9

        if ep % log_every == 0:
            print(f'Epoch {ep}: running return= {np.round(running_return, 2)}')                

        # store best actor
        if best_running_return < running_return:
            print(f"  best agent so far ({np.round(running_return, 2)})")
            best_running_return = running_return
            best_params = agent.state_dict().copy()

        optimizer.step()

    # restore best agent
    agent.load_state_dict(best_params)

def evaluate_agent(env, agent, n_episodes, max_steps_per_episode=1000):
    with torch.no_grad():
        mean_return = 0
        for ep in range(n_episodes):
            _, _, rewards = run_episode(env, agent, max_steps_per_episode=max_steps_per_episode)
            ep_reward = np.sum(rewards)
            mean_return += np.sum(rewards) / n_episodes

    return mean_return
