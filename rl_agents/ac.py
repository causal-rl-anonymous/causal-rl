import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from utils import print_log

eps = np.finfo(np.float32).eps.item()

############################################################

class ActorCritic(nn.Module):
    
    def __init__(self, s_nvals, a_nvals, hidden_size=32):

        super(ActorCritic, self).__init__()

        self.a_nvals = a_nvals
        self.s_nvals = s_nvals
        self.hidden_size = hidden_size

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.s_nvals, hidden_size),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.hidden_size, hidden_size),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.a_nvals),
            torch.nn.LogSoftmax(dim=-1),
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.s_nvals, hidden_size),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.hidden_size, hidden_size),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

def run_episode(env, model, max_steps_per_episode):

    action_log_probs, rewards, values = [], [], []

    with torch.no_grad():
        state = env.reset()

    for t in range(max_steps_per_episode):

        action_log_probs_t, value = model.forward(state)
        action = int(torch.multinomial(action_log_probs_t.exp(), 1)[0])

        with torch.no_grad():
            state, reward, done, info = env.step(action)

        action_log_prob = action_log_probs_t[:, action]

        action_log_probs.append(action_log_prob)
        values.append(value)
        rewards.append(reward)

        # if tmp_print_flag:
        #     action_desc = ["top", "right", "bottom", "left", "noop"]
        #     print(f"action={action_desc[action]} (p={torch.exp(action_log_prob).item()}), reward={reward}, value={value.detach().item()}")

        if done: 
            break

    return action_log_probs, values, rewards

def get_discounted_returns(rewards, gamma):
    
    n = len(rewards)
    returns = np.zeros(n)

    discounted_sum = 0.
    for t in range(n)[::-1]:
        discounted_sum = rewards[t] + gamma * discounted_sum
        returns[t] = discounted_sum

    return returns

def loss_episode(env, model, gamma, max_steps_per_episode):

    action_log_probs, values, rewards = run_episode(env, model, max_steps_per_episode)
    returns = get_discounted_returns(rewards, gamma)

    action_log_probs = torch.cat(action_log_probs, dim=-1).unsqueeze(0)
    values = torch.cat(values, dim=-1)
    returns = torch.tensor(returns, dtype=torch.float).unsqueeze(0)

    # compute actor-critic loss values
    actor_loss = - torch.sum(action_log_probs * (returns - values.detach()))
    critic_loss =  F.mse_loss(values, returns, reduction = 'sum')

    return actor_loss, critic_loss, np.sum(rewards)

def run_actorcritic(env, agent,
                    gamma=0.99,
                    n_epochs=10000,
                    batch_size=1,
                    max_steps_per_episode=1000,
                    log_every=1000,
                    lr=1e-2,
                    logfile=None):

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    best_running_return = -float("inf")
    best_params = agent.state_dict().copy()

    for ep in range(n_epochs) :

        epoch_return = 0
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        optimizer.zero_grad()

        for i in range(batch_size):
            actor_loss, critic_loss, episode_return = loss_episode(env, agent, gamma, max_steps_per_episode)

            epoch_return += episode_return / batch_size
            epoch_actor_loss += actor_loss.detach().item() / batch_size
            epoch_critic_loss += critic_loss.detach().item() / batch_size

            loss = (actor_loss + critic_loss) / batch_size
            loss.backward()

        if ep == 0:
            running_return = epoch_return
        else:
            running_return = epoch_return * 0.1 + running_return * 0.9

        if ep % log_every == 0:
            print_log(f'Epoch {ep}: running return= {np.round(running_return, 4)}, critic loss={np.round(epoch_critic_loss, 4)}', logfile=logfile)

        # store best agent
        if best_running_return < running_return:
            print_log(f"  best agent so far ({np.round(running_return, 4)})", logfile=logfile)
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
