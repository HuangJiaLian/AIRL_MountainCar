'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-11 19:17:36
@LastEditors: Jack Huang
@LastEditTime: 2019-11-15 19:04:03
'''

import tensorflow as tf 
import copy 

class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """
        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')
            # Actor
            # Input 20 20  act_space.n act_space.n 
            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                # 输出动作的概率
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space.n, activation=tf.nn.softmax)

            # Critic 
            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            # 辅助功能
            self.scope = tf.get_variable_scope().name

    def get_action(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    # Last Edit
    def get_distribution(self, obs):
        return tf.get_default_session().run(self.act_probs,feed_dict={self.obs: obs})

class PPO:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_theta_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # 待喂的漏斗
        self.assign_ops = []
        for v_old, v in zip(old_pi_trainable, pi_theta_trainable):
            self.assign_ops.append(tf.assign(v_old, v))
        
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
        self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
        self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        # construct computation graph for loss_clip
        # ratios = tf.divide(act_probs, act_probs_old)
        # 这样做是为了防止|ratios|为0，或者无穷大
        ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                      - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
        loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
        # reduce_mean相当于求和了
        loss_clip = tf.reduce_mean(loss_clip)


        # construct computation graph for loss of entropy bonus
        # Entropy = sum(P*logP)
        entropy = -tf.reduce_sum(self.Policy.act_probs *
                                 tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
        entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)


        # construct computation graph for loss of value function
        v_preds = self.Policy.v_preds
        loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
        loss_vf = tf.reduce_mean(loss_vf)

        # construct computation graph for loss
        loss = loss_clip - c_1 * loss_vf + c_2 * entropy

        # minimize -loss == maximize loss
        loss = -loss
        # 准备使用的优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_theta_trainable)
        # ???????????? 这里是应该使用pi_trainable还是 old_pi_trainable
        self.train_op = optimizer.minimize(loss, var_list=pi_theta_trainable)
        # 上面的部分都是定义了操作，还没有实际运算



    # lambda取的1， 对应的是MC,
    # 返回的每一个时刻的GAE  
    def get_gaes(self, rewards, v_preds, v_preds_next):
        # TD error: r_t + self.gamma * v_next - v 
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        # 先算最后一个时刻的GAE,再依次往前
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def train(self, obs, actions, gaes, rewards, v_preds_next):
            tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                                   self.Old_Policy.obs: obs,
                                                                   self.actions: actions,
                                                                   self.rewards: rewards,
                                                                   self.v_preds_next: v_preds_next,
                                                                   self.gaes: gaes})