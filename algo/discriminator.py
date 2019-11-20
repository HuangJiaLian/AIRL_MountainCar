'''
@Description: AIRL算法的Discriminator
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-11 19:18:07
@LastEditors: Jack Huang
@LastEditTime: 2019-11-18 17:36:07
'''

import tensorflow as tf 
import numpy as np 

class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        with tf.variable_scope('discriminator'):
            
            self.only_position = True
            if self.only_position:
                self.obs_t = tf.placeholder(tf.float32, shape=[None,1], name='obs') # s
                self.nobs_t = tf.placeholder(tf.float32, shape=[None,1], name='nobs') # s'
            else:
                self.obs_t = tf.placeholder(tf.float32, shape=[None] + list(env.observation_space.shape), name='obs') # s
                self.nobs_t = tf.placeholder(tf.float32, shape=[None] + list(env.observation_space.shape), name='nobs') # s'
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs') # log( \pi(a|s) )
                        
            self.gamma = 1.0
            self.score_discrim = True
            
            
            rew_input = self.obs_t
                        
            with tf.variable_scope('reward'):
                self.reward = self.reward_network(rew_input)
            
            with tf.variable_scope('value') as value_scope:
                h_ns = self.value_network(self.nobs_t)
                value_scope.reuse_variables()
                self.h_s = h_s = self.value_network(self.obs_t)
            
            log_f = self.reward + self.gamma*h_ns - h_s
            self.f_reward = tf.exp(log_f)
            log_p = self.lprobs

            log_fp = tf.reduce_logsumexp([log_f, log_p], axis=0)
            self.discrim_output = tf.exp(log_f-log_fp)

            with tf.variable_scope('loss'):
                self.loss = loss = -tf.reduce_mean(self.labels*(log_f-log_fp) + (1-self.labels)*(log_p-log_fp))

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.minimize(loss)

    # Reward approximator g
    def reward_network(self,reward_input):
        layer_1 = tf.layers.dense(inputs=reward_input, units=20, activation=tf.nn.leaky_relu, name='g_layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='g_layer2')
        out = tf.layers.dense(inputs=layer_2, units=1, name='g_out')
        return out

    # Value approximator h
    def value_network(self,value_input):
        layer_1 = tf.layers.dense(inputs=value_input, units=20, activation=tf.nn.leaky_relu, name='h_layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='h_layer2')
        out = tf.layers.dense(inputs=layer_2, units=1, name='h_out')
        return out

    def train(self, obs_t, nobs_t, lprobs, labels):
        return tf.get_default_session().run(self.train_op, feed_dict={self.obs_t: obs_t,
                                                                      self.nobs_t: nobs_t,
                                                                      self.lprobs: lprobs,
                                                                      self.labels: labels
                                                                      })
    # g reward
    def get_scores(self, obs_t):
        scores = tf.get_default_session().run(self.reward, feed_dict={self.obs_t: obs_t})
        return scores
    
    # f reward
    def get_f_scores(self):
        scores = tf.get_default_session().run(self.f_reward, feed_dict={self.obs_t: obs_t,
                                                                            self.nobs_t: kwargs['nobs_t']
                                                                            })
        return scores

    # log reward:logD-log(1-D)
    def get_l_scores(self, obs_t, nobs_t, lprobs):
        scores = tf.get_default_session().run(self.discrim_output, feed_dict={self.obs_t: obs_t,
                                                                            self.nobs_t: nobs_t,
                                                                            self.lprobs: lprobs
                                                                            })
        scores = np.log(scores) - np.log(1-scores)
        return scores

    

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)