from tensorflow.keras import models, layers, optimizers
import gym 
import numpy as np 
import os, shutil

# A very easy policy used for finding expert trajectories
def get_action(state):
    if state[-1] < 0:
        action = 0 # full throttle reverse
    elif state[-1] > 0:
        action = 2 # full throttle forward
    else:
        action = 1 # zero throttle
    return action


# Function used for saveing expert trajectories
def open_file_and_save(path, name, data):
    try:
        with open(os.path.join(path,name), 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(os.path.join(path,name), 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')



def main(env_name):
    # Initialise gym environment
    env = gym.make(env_name)
    ob_space = env.observation_space
    action_space = env.action_space.n 
    
    # Number of expert trajectories
    episodes = 20
    
    # The maximum steps in one episode
    max_steps = 200
    scores = []

    # Path for storing trajectories
    path = 'exp_traj'
    if os.path.exists(path) != True:
        os.makedirs(path)
    else:
        # Delete previous trajectories
        shutil.rmtree(path) 
        os.makedirs(path)
    
    for episode in range(episodes):
        state = env.reset()
        score = 0
        done = False 
        
        # Variables to collect data would be created in this episode
        observations = []
        actions = []
        returns = []
        for step in range(max_steps):

            # Uncomment rendering if run on a server
            # env.render()
            action = get_action(state=state)

            # Record states and actions
            observations.append(state)
            actions.append(action)

            # Interface with gym environment
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            if done:
                break
        # Record the return(the rewards sum of this episode)
        print('Return:', score)
        scores.append(score)    

        # Get the next states of current states
        next_observations = observations[1:]
        observations = observations[:-1]
        actions = actions[:-1]

        next_observations = np.reshape(next_observations, newshape=[-1] + list(ob_space.shape))
        observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        actions = np.array(actions).astype(dtype=np.int32)

        open_file_and_save(path = 'exp_traj', name = 'next_observations.csv', data = next_observations)
        open_file_and_save(path = 'exp_traj', name = 'observations.csv', data = observations)
        open_file_and_save(path = 'exp_traj', name = 'actions.csv', data = actions)

    
    scores = np.array(scores).astype(dtype=np.int32)
    open_file_and_save(path = 'exp_traj', name = 'returns.csv', data = scores)
    env.close()


if __name__ == "__main__":
    main(env_name="MountainCar-v0")