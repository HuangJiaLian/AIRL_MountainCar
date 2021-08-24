'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 19:27:08
@LastEditors: Jack Huang
@LastEditTime: 2021-08-24 19:24:52
'''

import gym 
import numpy as np 
import os, shutil

# A an policy used for finding expert trajectories
def get_action(state):
    # If moving to the left, full throttle reverse
    if state[-1] < 0:
        action = 0
    # If moving to the right, full throttle forward
    elif state[-1] > 0:
        action = 2 
    # Zero throttle
    else:
        action = 1 
    return action


# Saveing expert trajectories
def open_file_and_save(path, name, data):
    try:
        with open(os.path.join(path,name), 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(os.path.join(path,name), 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def main():
    # Initialise gym environment
    env = gym.make("MountainCar-v0")
    ob_space = env.observation_space
    action_space = env.action_space.n 
    
    # Number of expert trajectories
    episodes = 20
    
    # The maximum step limit in one episode to make sure the mountain car 
    # task is finite Markov decision processes (MDP).
    max_steps = 200
    
    # A list to store the
    # return, (i.e, the sum of the reward in an episode),in each episode.
    return_Gs = []

    # Create path for storing trajectories
    path = 'exp_traj'
    if os.path.exists(path) != True:
        os.makedirs(path)
    else:
        # Delete previous trajectories before creating a new one
        shutil.rmtree(path) 
        os.makedirs(path)
    
    for episode in range(episodes):
        print('Episode:{}'.format(episode+1))
        # Get the initial state of environment
        state = env.reset()
        return_G = 0
        done = False 
        
        # Variables to collect data would be created in this episode
        observations = []
        actions = []
        for step in range(max_steps):
            # env.render() # Uncomment rendering if run on a server 
            # Get action of expert according to the new state
            action = get_action(state=state)

            # Record states and actions
            observations.append(state)
            actions.append(action)

            # Interact with gym environment, and 
            # obtain a new reward and a new state
            next_state, next_reward, done, _ = env.step(action)
            return_G += next_reward
            state = next_state

            # If the car reach the goal, end the episode.
            if done:
                break
        
        # Record the return(the rewards sum of this episode)
        print('Return:{}'.format(return_G))
        return_Gs.append(return_G)    

        # The next states, S_{t+1}
        next_observations = observations[1:]
        observations = observations[:-1]
        actions = actions[:-1]

        # Save as csv files
        next_observations = np.reshape(next_observations, newshape=[-1] + list(ob_space.shape))
        observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        actions = np.array(actions).astype(dtype=np.int32)
        open_file_and_save(path = 'exp_traj', name = 'next_observations.csv', data = next_observations)
        open_file_and_save(path = 'exp_traj', name = 'observations.csv', data = observations)
        open_file_and_save(path = 'exp_traj', name = 'actions.csv', data = actions)
    
    # Save returns 
    return_Gs = np.array(return_Gs).astype(dtype=np.int32)
    open_file_and_save(path = 'exp_traj', name = 'returns.csv', data = return_Gs)
    env.close()
    print('Done: {} expert trajectories obtained.'.format(episodes))


if __name__ == "__main__":
    main()