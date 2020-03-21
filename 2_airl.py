'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 19:27:08
@LastEditors: Jack Huang
@LastEditTime: 2019-11-18 19:24:52
'''

import tensorflow as tf 
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
    # Env 
    env = gym.make('MountainCar-v0') 
    ob_space = env.observation_space
    
    # For Reinforcement Learning
    Policy = gen.Policy_net('policy', env)
    Old_Policy = gen.Policy_net('old_policy', env)
    PPO = gen.PPO(Policy, Old_Policy, gamma=0.95)
    
    # For Inverse Reinforcement Learning
    D = dis.Discriminator(env)
    
    # Load Experts Demonstration
    expert_observations = np.genfromtxt('exp_traj/observations.csv')
    next_expert_observations = np.genfromtxt('exp_traj/next_observations.csv')
    expert_actions = np.genfromtxt('exp_traj/actions.csv', dtype=np.int32)

    expert_returns = np.genfromtxt('exp_traj/returns.csv')
    mean_expert_return = np.mean(expert_returns)
    
    max_episode = 24000
    max_steps = 200
    saveReturnEvery = 100
    num_expert_tra = 20 

    # Logger 用来记录训练过程
    train_logger = log.logger(logger_name='AIRL_MCarV0_Training_Log', 
        logger_path='./trainingLog/', col_names=['Episode', 'Actor(D)', 'Expert Mean(D)','Actor','Expert Mean'])
    
    # Saver to save all the variables
    model_save_path = './model/'
    model_name = 'airl'
    saver = tf.train.Saver(max_to_keep=int(max_episode/100))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.reset() 
        # do NOT use original rewards to update policy
        for episode in range(max_episode):
            if episode % 100 == 0:
                print('Episode ', episode)
            
            observations = []
            actions = []
            rewards = []
            v_preds = []

            # 遍历这次游戏中的每一步
            obs = env.reset()
            for step in range(max_steps):
                # if episode % 100 == 0:
                #     env.render()
                obs = np.stack([obs]).astype(dtype=np.float32)
                act, v_pred = Policy.get_action(obs=obs, stochastic=True)
                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                # 和环境交互
                next_obs, reward, done, info = env.step(act)

                observations.append(obs)
                actions.append(act)
                # 这里的reward并不是用来更新网络的,而是用来记录真实的
                # 表现的。
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    break
                else:
                    obs = next_obs

            # 完了就可以用数据来训练网络了

            # 准备数据
            # Expert的数据已经准备好了
            # Generator的数据
            # convert list to numpy array for feeding tf.placeholder

            next_observations = observations[1:]
            observations = observations[:-1]
            actions = actions[:-1]

            next_observations = np.reshape(next_observations, newshape=[-1] + list(ob_space.shape))
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            # Get the G's probabilities 
            probabilities = get_probabilities(policy=Policy, observations=observations, actions=actions)
            # Get the experts' probabilities
            expert_probabilities = get_probabilities(policy=Policy, observations=expert_observations, actions=expert_actions)
            
            # numpy 里面log的底数是e
            log_probabilities = np.log(probabilities)
            log_expert_probabilities = np.log(expert_probabilities)
            
            if D.only_position:
                observations_for_d = (observations[:,0]).reshape(-1,1)
                next_observations_for_d = (next_observations[:,0]).reshape(-1,1)
                expert_observations_for_d = (expert_observations[:,0]).reshape(-1,1)
                next_expert_observations_for_d = (next_expert_observations[:,0]).reshape(-1,1)
            log_probabilities_for_d = log_probabilities.reshape(-1,1)
            log_expert_probabilities_for_d = log_expert_probabilities.reshape(-1,1)

            # 数据排整齐
            
            obs, obs_next, acts, path_probs = \
                observations_for_d, next_observations_for_d, actions, log_probabilities
            expert_obs, expert_obs_next, expert_acts, expert_probs = \
                expert_observations_for_d, next_expert_observations_for_d, expert_actions, log_expert_probabilities
            
            acts = acts.reshape(-1,1)
            expert_acts = expert_acts.reshape(-1,1)

            path_probs = path_probs.reshape(-1,1)
            expert_probs = expert_probs.reshape(-1,1)
            
            # train discriminator 得到Reward函数
            # print('Train D')
            # 这里两类数据量的大小不对等啊
            # 应该可以优化的
            batch_size = 32
            for i in range(1):
                # 抽一个G的batch
                nobs_batch, obs_batch, act_batch, lprobs_batch = \
                    sample_batch(obs_next, obs, acts, path_probs, batch_size=batch_size)
                
                # 抽一个Expert的batch
                nexpert_obs_batch, expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                    sample_batch(expert_obs_next, expert_obs, expert_acts, expert_probs, batch_size=batch_size)
                
                # 前半部分负样本0; 后半部分是正样本1
                labels = np.zeros((batch_size*2, 1))
                labels[batch_size:] = 1.0

                # 拼在一起喂到神经网络里面去训练
                obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
                nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
                # 若是只和状态相关，下面这个这个没有用
                act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
                lprobs_batch = np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0)
                D.train(obs_t = obs_batch, 
                        nobs_t = nobs_batch, 
                        lprobs = lprobs_batch, 
                        labels = labels)
            if episode % 50 == 0:
                drawRewards(D=D, episode=episode, path='./trainingLog/')

            # output of this discriminator is reward
            if D.score_discrim == False:
                d_rewards = D.get_scores(obs_t=observations_for_d)
            else:
                d_rewards = D.get_l_scores(obs_t=observations_for_d, nobs_t=next_observations_for_d, lprobs=log_probabilities_for_d)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
            d_actor_return = np.sum(d_rewards)
            # print(d_actor_return)

            # d_expert_return: Just For Tracking
            if D.score_discrim == False:
                expert_d_rewards = D.get_scores(obs_t=expert_observations_for_d)
            else:
                expert_d_rewards = D.get_l_scores(obs_t=expert_observations_for_d, nobs_t= next_expert_observations_for_d,lprobs= log_expert_probabilities_for_d )
            expert_d_rewards = np.reshape(expert_d_rewards, newshape=[-1]).astype(dtype=np.float32)
            d_expert_return = np.sum(expert_d_rewards)/num_expert_tra
            # print(d_expert_return)

            ######################
            # Start Logging      #
            ######################
            train_logger.add_row_data([episode, d_actor_return, d_expert_return, 
                                sum(rewards), mean_expert_return], saveFlag=True)
            if episode % saveReturnEvery == 0:
                train_logger.plotToFile(title='Return')
            ###################
            # End logging     # 
            ###################

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy 得到更好的Policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]

            # if episode % 4 == 0:
            #     PPO.assign_policy_parameters()
            
            PPO.assign_policy_parameters()


            for epoch in range(10):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])
            # 保存整个模型
            if episode > 0 and episode % 100 == 0:
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=episode)

if __name__ == '__main__':
    main()
