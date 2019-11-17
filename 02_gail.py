'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 19:27:08
@LastEditors: Jack Huang
@LastEditTime: 2019-10-11 19:32:55
'''

import tensorflow as tf 
import numpy as np 
import gym, os
import algo.generator as gen 
import algo.discriminator as dis 
import utility.logger as log 


def main():
    # 环境
    env = gym.make('MountainCar-v0') 
    ob_space = env.observation_space
    
    # RL部分
    Policy = gen.Policy_net('policy', env)
    Old_Policy = gen.Policy_net('old_policy', env)
    PPO = gen.PPO(Policy, Old_Policy, gamma=0.95)
    
    # IRL部分
    D = dis.Discriminator(env)
    
    # 加载Expert
    # Numpy 也是可以读txt然后直接保存成数字的
    expert_observations = np.genfromtxt('exp_traj/observations.csv')
    expert_actions = np.genfromtxt('exp_traj/actions.csv', dtype=np.int32)
    expert_returns = np.genfromtxt('exp_traj/returns.csv')
    mean_expert_return = np.mean(expert_returns)
    
    max_episode = 12000
    max_steps = 200
    saveReturnEvery = 100
    num_expert_tra = 20 

    # Logger 用来记录训练过程
    train_logger = log.logger(logger_name='MCarV0_Training_Log', 
        logger_path='./trainingLog/', col_names=['Episode', 'Actor(D)', 'Expert Mean(D)','Actor','Expert Mean'])
    
    # Saver to save all the variables
    model_save_path = './model/'
    model_name = 'gail'
    saver = tf.train.Saver(max_to_keep=120)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.reset() 
        # do NOT use rewards to update policy
        # 玩很多次游戏
        for episode in range(max_episode):
            if episode % 100 == 0:
                print('Episode ', episode)
            # 开始玩每把游戏前，准备几个管子，用来收集过程中遇到的东西
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
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            # print('Generator Data:')
            # print(observations)
            # input()
            # print(actions)
            # input()

            # train discriminator 得到Reward函数
            # print('Train D')
            # 这里两类数据量的大小不对等啊
            # 应该可以优化的
            for i in range(2):
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # output of this discriminator is reward
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
            d_actor_return = np.sum(d_rewards)
            # print(d_actor_return)

            # d_expert_return: Just For Tracking
            expert_d_rewards = D.get_rewards(expert_observations, expert_actions)
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


            for epoch in range(6):
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