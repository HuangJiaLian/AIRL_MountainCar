'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 19:27:08
@LastEditors: Jack Huang
@LastEditTime: 2019-11-19 15:21:38
'''

import tensorflow as tf 
import numpy as np 
import gym, os
import argparse
import algo.generator as gen 
import algo.discriminator as dis 
import utility.logger as log
import matplotlib.pyplot as plt  


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

    # 测试的时候可以少一点步数
    max_episode = 1000
    max_steps = 200
    saveReturnEvery = 100
    num_expert_tra = 20 
    
    # Saver to save all the variables
    model_save_path = './modelGAN/'
    model_name = 'airl'
    saver = tf.train.Saver() 
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    if ckpt and ckpt.model_checkpoint_path:
            print('Found Saved Model.')
            # -1 代表最新的
            # ckpt_to_restore = ckpt.all_model_checkpoint_paths[-1]
            ckpt_to_restore = ckpt.all_model_checkpoint_paths[-1]
    else:
        print('No Saved Model. Exiting')
        exit()
    
    # Logger 用来记录训练过程
    train_logger = log.logger(logger_name='MCarV0_Training_Log', 
        logger_path='./testLog_' + ckpt_to_restore.split('-')[-1] + '/', \
            col_names=['Episode', 'Actor(D)', 'Expert Mean(D)','Actor','Expert Mean'])
    


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Restore Model
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt_to_restore)
            print('Model Restored.')


        
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
                env.render()
                obs = np.stack([obs]).astype(dtype=np.float32)

                # When testing set stochastic False will get better performance
                # act, v_pred = Policy.get_action(obs=obs, stochastic=True)
                act, v_pred = Policy.get_action(obs=obs, stochastic=False)
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
            print('Rewards: ', sum(rewards))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', nargs='?', const=-1, type=int, default=-1, \
                        help="choose the num(th) checkpoint to restore.")
    args = parser.parse_args()
    print(args.N)
    main()