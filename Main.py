import gym
import numpy as np
import time
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create environment, Note: Used human render mode but to test probably swap to 'rgb_array'
env = gym.make('CartPole-v1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Reset before test and return a tuple of the current state and any info which i am ignoring with _
# Note: states to be observed are cart pos, cart vel, pole angle and pole angular vel
(state,_) = env.reset()
rewards_per_episode = []

# All hyper parameters
episodes = 1000000
batch_size = 64
LearnRate = 0.001
memlen = 1000
epsilonInit = 1
eps_final = 0.01
gamma = 0.999
update_interval = 10


# neural network for agent
class DQN(nn.Module): # inherit from module class (all nn classes must)
    def __init__(self, state_dim, action_dim): # Note: action dim is 2 because it can only go left or right, state dim is 4 for similar reasons
        super(DQN,self).__init__() # inherit from DQN
        self.layer1 = nn.Linear(state_dim,128) # each layer done individually, this could be done with nn.Sequential but each layer allows more control
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,action_dim)
        

    def forward(self,x): # forward class function for forward propagation (forward just returns the value by the NN)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
# --- Replay Buffer ---
class ReplayMemory:
    def __init__(self, maxLen):
        self.memory = deque([], maxlen=maxLen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sampleSize):
        batch = random.sample(self.memory, sampleSize)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.memory)
    
# Epsilon greedy action selection
def select_action(state, Q_network, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            qvals = Q_network(state)
            return torch.argmax(qvals).item()

def train():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim,action_dim).to(device)
    target_net = DQN(state_dim,action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimiser = optim.Adam(policy_net.parameters(), lr=LearnRate)
    memory = ReplayMemory(memlen)

    epsilon = epsilonInit

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False

        while not done:
            action = select_action(state,policy_net,epsilon)
            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                states,actions,rewards,next_states,dones = memory.sample(batch_size)

                with torch.no_grad():
                    next_q = target_net(next_states).max(1,keepdim=True)[0]
                    target_q = rewards + gamma * next_q *(1 - dones)

                current_q = policy_net(states).gather(1, actions)

                loss = nn.MSELoss()(current_q, target_q)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        epsilon = max(eps_final, epsilon * 0.9999)
        rewards_per_episode.append(total_reward)

        if episode % update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

        # Early stopping condition
        if total_reward >= 100000:
            print("Environment solved!")
            break

    env.close()

    # Save the trained model
    #torch.save(policy_net.state_dict(), 'dqn_cartpoleV1.pth')
    #print("Model saved as dqn_cartpoleV1.pth")


train()
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN on CartPole-v1')
plt.grid()
plt.show()



    

        



        



    
