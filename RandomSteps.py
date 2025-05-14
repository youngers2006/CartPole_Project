import gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Create environment, Note: Used human render mode but to test probably swap to 'rgb_array'
env = gym.make('CartPole-v1',render_mode='human')
# Reset before test and return a tuple of the current state and any info which i am ignoring with _
# Note: states to be observed are cart pos, cart vel, pole angle and pole angular vel
(state,_) = env.reset()

env.render() # Render current step
env.step(0) # Returns output tuple

env.observation_space
env.observation_space.high
env.observation_space.low
env.action_space
env.spec
env.spec.max_episode_steps
env.spec.reward_threshold 

episodeNumber = 5
timeSteps = 100

for episodeIndex in range(episodeNumber):
    init_state = env.reset()
    print(episodeIndex)
    env.render()
    appendedObservation=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
        appendedObservation.append(observation)
        time.sleep(0.1)
        if (terminated):
            print(f"Episode {episodeIndex + 1} ended after {timeIndex + 1} steps.")
            time.sleep(1)
            break

env.close()





#torch.save(model.state_dict(), 'cartpole_model.pth')  # Save the model
#print("Model saved to 'cartpole_model.pth'")



