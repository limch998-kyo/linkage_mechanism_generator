import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Neural Network model for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500

# Training function
def train(model, memory, optimizer, target_model):
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    current_q = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
    next_q = target_model(next_state_batch).max(1)[0]
    expected_q = reward_batch + GAMMA * next_q * (1 - done_batch)
    
    loss = nn.MSELoss()(current_q, expected_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main function
if __name__ == '__main__':
    import random

    env = gym.make('CartPole-v1')
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    model = DQN(n_state, n_action)
    target_model = DQN(n_state, n_action)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), LR)
    memory = []

    n_episodes = 300
    steps_done = 0

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(1000):
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action = model(torch.FloatTensor(state)).argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _, _ = env.step(action)

            memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            train(model, memory, optimizer, target_model)

            if done:
                break
        
        # Update target model every few episodes
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        print(f'Episode {episode}, Total Reward: {total_reward}')

    env.close()
