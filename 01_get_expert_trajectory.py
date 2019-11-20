'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 17:30:07
@LastEditors: Jack Huang
@LastEditTime: 2019-11-18 19:23:52
'''
# hhhh
from tensorflow.keras import models, layers, optimizers
import gym 
import numpy as np 

# 专家策略
def get_action(state):
    if state[-1] < 0:
        action = 0 # 向左
    elif state[-1] > 0:
        action = 2 # 向右
    else:
        action = 1
    return action


# 用来保存专家轨迹用的
def open_file_and_save(file_path, data):
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')



def main(env_name):
    # 准备环境
    env = gym.make(env_name)
    ob_space = env.observation_space
    action_space = env.action_space.n 
    # 表示玩episodes次游戏, 拿这么多条专家轨迹
    episodes = 20
    # 每一轮游戏里规定的最大步数
    max_steps = 200
    scores = []

    for episode in range(episodes):
        state = env.reset()
        score = 0
        done = False 
        
        # 开始玩每把游戏前，准备几个管子，用来收集过程中遇到的东西
        observations = []
        actions = []
        returns = []
        for step in range(max_steps):
            # env.render()
            action = get_action(state=state)
            # 记录状态
            observations.append(state)
            actions.append(action)

            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            if done:
                break
        
        print('Return:', score)
        scores.append(score)    

        # 这样做都是为了s'
        next_observations = observations[1:]
        observations = observations[:-1]
        actions = actions[:-1]

        next_observations = np.reshape(next_observations, newshape=[-1] + list(ob_space.shape))
        observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        actions = np.array(actions).astype(dtype=np.int32)

        open_file_and_save('exp_traj/next_observations.csv', next_observations)
        open_file_and_save('exp_traj/observations.csv', observations)
        open_file_and_save('exp_traj/actions.csv', actions)

    
    scores = np.array(scores).astype(dtype=np.int32)
    open_file_and_save('exp_traj/returns.csv', scores)
    env.close()


if __name__ == "__main__":
    main(env_name="MountainCar-v0")