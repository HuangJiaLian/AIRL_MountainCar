'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 19:27:08
@LastEditors: Jack Huang
@LastEditTime: 2021-08-24 19:24:52
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np 
import gym, os
import algo.generator as gen 
import algo.discriminator as dis 
import utility.logger as log 
import matplotlib.pyplot as plt 

def get_probabilities(policy, observations, actions):
    # Evaluate distribution
    distributions = policy.get_distribution(observations)
    # Fancy Index to get probabilities
    probabilities = distributions[np.arange(distributions.shape[0]), actions]
    return probabilities

def sample_batch(*args, batch_size=32):
    N = args[0].shape[0]
    batch_idxs = np.random.randint(0, N, batch_size)
    return [data[batch_idxs] for data in args]

def drawRewards(D, episode, path):
    plt.clf()
    plt.xlabel('x_position')
    plt.ylabel('reward')
    # Draw reward function
    # Prepare x positions
    x_positions = (np.linspace(-1.2, 0.6, 100)).reshape(-1,1)
    # Get rewards 
    all_rewards = D.get_scores(obs_t=x_positions)
    # Plot
    plt.plot(x_positions,all_rewards)
    plt.savefig(os.path.join(path, str(episode) + '_learned_rewards.png'))
    plt.clf()

def main():
    # Mountain care env setting
    env = gym.make('MountainCar-v0') 
    ob_space = env.observation_space
    action_space = env.action_space
    print(ob_space, action_space)
    
    # For Reinforcement Learning
    Policy = gen.Policy_net('policy', env)
    Old_Policy = gen.Policy_net('old_policy', env)
    PPO = gen.PPO(Policy, Old_Policy, gamma=0.95)
    
    # For Inverse Reinforcement Learning
    D = dis.Discriminator(env)
    
    # Load expert trajectories
    expert_observations = np.genfromtxt('exp_traj/observations.csv')
    next_expert_observations = np.genfromtxt('exp_traj/next_observations.csv')
    expert_actions = np.genfromtxt('exp_traj/actions.csv', dtype=np.int32)
    # Expert returns is just used for showing the mean scrore, not for training
    expert_returns = np.genfromtxt('exp_traj/returns.csv')
    mean_expert_return = np.mean(expert_returns)
    
    max_episode = 10000
    # The maximum step limit in one episode to make sure the mountain car 
    # task is finite Markov decision processes (MDP).
    max_steps = 200
    saveReturnEvery = 100
    num_expert_tra = 20 

    # Just use to record the training process
    train_logger = log.logger(logger_name='AIRL_MCarV0_Training_Log', 
        logger_path='./trainingLog/', col_names=['Episode', 'Actor(D)', 
        'Expert Mean(D)','Actor','Expert Mean'])
    
    # Model saver
    model_save_path = './model/'
    model_name = 'airl'
    saver = tf.train.Saver(max_to_keep=int(max_episode/100))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(max_episode):
            if episode % 100 == 0:
                print('Episode ', episode)
            observations = []
            actions = []
            rewards = []
            v_preds = []

            obs = env.reset()
            # Interact with the environment until reach
            # the terminal state or the maximum step.
            for step in range(max_steps):
                # if episode % 100 == 0:
                #     env.render()

                obs = np.stack([obs]).astype(dtype=np.float32)
                # act, v_pred = Policy.get_action(obs=obs, stochastic=True)
                act, v_pred = Old_Policy.get_action(obs=obs, stochastic=True)
                

                next_obs, reward, done, _ = env.step(act)

                observations.append(obs)
                actions.append(act)
                # DO NOT use original rewards to update policy
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    # next state of terminate state has 0 state value
                    v_preds_next = v_preds[1:] + [0]  
                    break
                else:
                    obs = next_obs

            # Data preparation
            # Data for generator: convert list to numpy array for feeding tf.placeholder
            next_observations = observations[1:]
            observations = observations[:-1]
            actions = actions[:-1]

            next_observations = np.reshape(next_observations, newshape=[-1] + list(ob_space.shape))
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            # G's probabilities 
            probabilities = get_probabilities(policy=Policy, \
                observations=observations, actions=actions)
            # Experts' probabilities
            expert_probabilities = get_probabilities(policy=Policy, \
                observations=expert_observations, actions=expert_actions)
            
            log_probabilities = np.log(probabilities)
            log_expert_probabilities = np.log(expert_probabilities)
            
            # Prepare data for disriminator
            if D.only_position:
                observations_for_d = (observations[:,0]).reshape(-1,1)
                next_observations_for_d = (next_observations[:,0]).reshape(-1,1)
                expert_observations_for_d = (expert_observations[:,0]).reshape(-1,1)
                next_expert_observations_for_d = (next_expert_observations[:,0]).reshape(-1,1)
            log_probabilities_for_d = log_probabilities.reshape(-1,1)
            log_expert_probabilities_for_d = log_expert_probabilities.reshape(-1,1)

            
            obs, obs_next, acts, path_probs = \
                observations_for_d, next_observations_for_d, \
                actions.reshape(-1,1), log_probabilities.reshape(-1,1)
            expert_obs, expert_obs_next, expert_acts, expert_probs = \
                expert_observations_for_d, next_expert_observations_for_d, \
                expert_actions.reshape(-1,1), log_expert_probabilities.reshape(-1,1)
        
            
            
            # 这里两类数据量的大小不对等啊, 应该可以优化的??
            # Train discriminator
            batch_size = 32
            for i in range(1):
                # Sample generator
                nobs_batch, obs_batch, act_batch, lprobs_batch = \
                    sample_batch(obs_next, obs, acts, path_probs, batch_size=batch_size)
                # Sample expert
                nexpert_obs_batch, expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                    sample_batch(expert_obs_next, expert_obs, expert_acts, \
                    expert_probs, batch_size=batch_size)
                
                # Label generator samples as 0, indicating that discriminator 
                # always consider generator's behavior is not good;
                # Label expert samples as 1, indicating that discriminator 
                # always consider expert's behavior is excellent.
                labels = np.zeros((batch_size*2, 1))
                labels[batch_size:] = 1.0

                obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
                nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
                lprobs_batch = np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0)
                D.train(obs_t = obs_batch, 
                        nobs_t = nobs_batch, 
                        lprobs = lprobs_batch, 
                        labels = labels)
            
            if episode % 50 == 0:
                drawRewards(D=D, episode=episode, path='./trainingLog/')

            # The output of this discriminator is reward
            if D.score_discrim == False:
                d_rewards = D.get_scores(obs_t=observations_for_d)
            else:
                d_rewards = D.get_l_scores(obs_t=observations_for_d, \
                    nobs_t=next_observations_for_d, lprobs=log_probabilities_for_d)
            
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
            # Sum rewards to get return: Just for tracking the record of returns overtime.  
            d_actor_return = np.sum(d_rewards) 

            if D.score_discrim == False:
                expert_d_rewards = D.get_scores(obs_t=expert_observations_for_d)
            else:
                expert_d_rewards = D.get_l_scores(obs_t=expert_observations_for_d, \
                    nobs_t= next_expert_observations_for_d,lprobs= log_expert_probabilities_for_d )
            expert_d_rewards = np.reshape(expert_d_rewards, newshape=[-1]).astype(dtype=np.float32)
            d_expert_return = np.sum(expert_d_rewards)/num_expert_tra
 
            #** Start Logging **#: Just use to track information
            train_logger.add_row_data([episode, d_actor_return, d_expert_return, 
                                sum(rewards), mean_expert_return], saveFlag=True)
            if episode % saveReturnEvery == 0:
                train_logger.plotToFile(title='Return')
            #** End logging  **# 

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            inp = [observations, actions, gaes, d_rewards, v_preds_next]

            if episode % 4 == 0:
                PPO.assign_policy_parameters()
            # PPO.assign_policy_parameters()

            for epoch in range(10):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],size=32)
                # sample training data
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])
                          
            # Save model
            if episode > 0 and episode % 100 == 0:
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=episode)

if __name__ == '__main__':
    main()
