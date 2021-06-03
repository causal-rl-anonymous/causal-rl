import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class Actor(torch.nn.Module):

    def __init__(self, s_nvals, a_nvals, hidden_size=32):
        super().__init__()

        self.s_nvals = s_nvals
        self.a_nvals = a_nvals

        self.log_q_a_s = torch.nn.Sequential(
            torch.nn.Linear(self.s_nvals, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.a_nvals),
            torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, state):
        return self.log_q_a_s(state)

    def get_action(self, state, with_log_prob=False):
        log_q_a_s = self.log_q_a_s(state)
        action = torch.distributions.categorical.Categorical(logits=log_q_a_s).sample()
        if with_log_prob:
            log_prob = log_q_a_s[:, action]
            return action, log_prob
        else:
            return action

def reinforce_loss(action_log_probs, rewards, gamma):

    assert gamma > 0 and gamma <= 1

    # cumulated discounted rewards
    returns = rewards.float().clone()
    for t in range(1, len(rewards)):
        returns[:, :-t] += gamma**t * rewards[:, t:]

    # return normalization
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # policy gradient loss
    loss = (-action_log_probs * returns).sum(dim=1).mean(dim=0)

    return loss

def run_reinforce(env, actor, lr, batch_size=1, gamma=0.99, n_epochs=500, log_every=10):

    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        for i in range(batch_size):
            state = env.reset()
            done = False
            log_probs = []
            rewards = []

            while not done:
                state = state.float().detach()
                action, log_prob = actor.get_action(state, with_log_prob=True)
                state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

            log_probs = torch.cat(log_probs, dim=1)
            rewards = torch.tensor(rewards, dtype=float).unsqueeze(0)

            loss = reinforce_loss(log_probs, rewards, gamma=gamma) / batch_size

            loss.backward()

        optimizer.step()

        epoch_return = rewards.sum(1).mean(0)

        if epoch == 0:
            running_return = epoch_return
        else:
            running_return = 0.9 * running_return + 0.1 * epoch_return

        if epoch % log_every == 0:
            print(f"epoch {epoch} return {running_return}")
