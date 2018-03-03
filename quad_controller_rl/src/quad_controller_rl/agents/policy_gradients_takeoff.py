"""Policy gradient agent."""
import os

import numpy as np
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent

import tensorflow as tf

class ReplayBuffer:

    def __init__(self, maxlen, action_shape, state_shape, dtype=np.float32):
        """Initialize a ReplayBuffer object."""
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.state_data = np.zeros((maxlen,) + state_shape).astype(dtype)
        self.action_data = np.zeros((maxlen,) + action_shape).astype(dtype)
        self.reward_data = np.zeros((maxlen,1)).astype(dtype)
        self.next_state_data = np.zeros((maxlen,) + state_shape).astype(dtype)
        self.done_data = np.zeros((maxlen,1)).astype(dtype)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            self.length += 1
        idx = (self.start + self.length - 1) % self.maxlen
        self.state_data[idx] = state
        self.action_data[idx] = action
        self.reward_data[idx] = reward
        self.next_state_data[idx] = next_state
        self.done_data[idx] = done
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.random_integers(self.length - 1, size=batch_size)
        sampled = {'states':self.set_min_ndim(self.state_data[idxs]),
                   'actions':self.set_min_ndim(self.action_data[idxs]),
                   'rewards':self.set_min_ndim(self.reward_data[idxs]),
                   'next_states':self.set_min_ndim(self.next_state_data[idxs]),
                   'dones':self.set_min_ndim(self.done_data[idxs])}
        return sampled
    
    def set_min_ndim(self,x):
        """set numpy array minimum dim to 2 (for sampling)"""
        if x.ndim < 2:
            return x.reshape(-1,1)
        else:
            return x

    def __len__(self):
        return self.length
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    #0.15 0.3
    def __init__(self, size, mu=None, theta=0.15, sigma=0.02, dt=1e-2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class A2C:
    def __init__(self, state_shape, action_shape, actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        tf.reset_default_graph()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.nb_actions = np.prod(self.action_shape)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        
        #inputs
        self.input_state = tf.placeholder(tf.float32, (None,) + self.state_shape, name='input_state')
        self.input_action = tf.placeholder(tf.float32, (None,) + self.action_shape, name='input_action')
        self.input_state_target = tf.placeholder(tf.float32, (None,) + self.state_shape, name='input_state_target')
        self.rewards = tf.placeholder(tf.float32, (None,1), name='rewards')
        self.dones =tf.placeholder(tf.float32, (None,1), name='dones')
        
        #local and target nets
        self.actor = self.actor_net(self.input_state, self.nb_actions,name='actor')
        self.critic = self.critic_net(self.input_state, self.input_action,name='critic')
        self.actor_and_critic = self.critic_net(self.input_state,self.actor,name='critic',reuse=True)
        
        self.actor_target = self.actor_net(self.input_state_target, self.nb_actions, name='target_actor')
        self.actor_and_critic_target = self.critic_net(self.input_state_target,
                                                       self.actor_target, name='target_critic')
        
        self.actor_loss, self.critic_loss = self.set_model_loss(self.critic, self.actor_and_critic,
                                                                self.actor_target, self.actor_and_critic_target,
                                                                self.rewards, self.dones, self.gamma)
        
        self.actor_opt, self.critic_opt = self.set_model_opt(self.actor_loss, self.critic_loss,
                                                             self.actor_lr, self.critic_lr)
        
    
    def actor_net(self, state, nb_actions, name, reuse=False, training=True):
        with tf.variable_scope(name, reuse=reuse):
            x = tf.layers.Dense(64,activation=tf.nn.relu)(state)
            x = tf.layers.Dense(64,activation=tf.nn.relu)(x)
            actions = tf.layers.Dense(nb_actions,
                                      activation=tf.tanh,
                                      kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(x)
            return actions
    
    def critic_net(self, state, action, name, reuse=False, training=True):
        with tf.variable_scope(name, reuse=reuse):
            x = tf.layers.Dense(64,activation=tf.nn.relu)(state)
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.Dense(64,activation=tf.nn.relu)(x)
            q = tf.layers.Dense(1,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(x)
            return q
    
    def set_model_loss(self, critic, actor_and_critic, actor_target, actor_and_critic_target, rewards, dones, gamma):
        Q_targets = rewards + (gamma * actor_and_critic_target) * (1. - dones)
        actor_loss = tf.reduce_mean(-actor_and_critic)
        tf.losses.add_loss(actor_loss)
        critic_loss = tf.losses.huber_loss(Q_targets,critic)
        return actor_loss, critic_loss
    
    def set_model_opt(self, actor_loss, critic_loss, actor_lr, critic_lr):
        train_vars = tf.trainable_variables()
        actor_vars = [var for var in train_vars if var.name.startswith('actor')]
        critic_vars = [var for var in train_vars if var.name.startswith('critic')]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            actor_opt = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss, var_list=actor_vars)
            critic_opt = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss, var_list=critic_vars)
        return actor_opt, critic_opt
    
        
class DDPGtakeoff(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task  # should contain observation_space and action_space
        self.state_shape = (7,)
        self.action_shape = (1,)
        self.nb_actions = np.prod(self.action_shape)
        self.action_range = self.task.action_space.high[2]-self.task.action_space.low[2]
        
        # Replay memory
        self.buffer_size = 100000
        self.batch_size =128
        self.memory = ReplayBuffer(self.buffer_size,self.action_shape, self.state_shape)
        
        # Noise process
        self.noise = OUNoise(self.nb_actions)
        
        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005 # 0.005
        self.actor_lr = 0.0001  #0.0001
        self.critic_lr = 0.001
        
        #initialize
        self.a2c = A2C(self.state_shape, self.action_shape, actor_lr=self.actor_lr, critic_lr=self.critic_lr,
                       gamma=self.gamma)
        self.initialize()
        
        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]
        #initial episode vars
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.acts = np.zeros(shape=self.task.action_space.shape)
        
    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only
        
    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.acts = np.zeros(shape=self.task.action_space.shape)
        
    def step(self, state, reward, done):
        action = self.act(state[None,:])
        self.count += 1
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.memory.add(self.last_state, self.last_action, reward, state, done)
        if (len(self.memory) > self.batch_size):
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        self.last_state = state
        self.last_action = action
        self.acts[2] = action * self.action_range * 0.5
        if done:
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            print('total reward={:7.4f}, count={}'.format(self.total_reward,self.count))
            self.reset_episode_vars()
        return self.acts

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        actions = self.sess.run(self.a2c.actor, feed_dict={self.a2c.input_state:states})
        return actions + self.noise.sample()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        states = experiences['states']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_states = experiences['next_states']
        dones = experiences['dones']
        
        #actor critic update
        self.sess.run([self.a2c.actor_opt,self.a2c.critic_opt],feed_dict={self.a2c.input_state:states,
                                                                              self.a2c.input_action:actions,
                                                                              self.a2c.input_state_target:next_states,
                                                                              self.a2c.rewards:rewards,
                                                                              self.a2c.dones:dones})
        #target soft update
        self.sess.run(self.soft_update_ops)
        
        
    def initialize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        actor_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        actor_target_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')
        critic_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        critic_target_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')
        target_init_ops = []
        soft_update_ops = []
        for var, target_var in zip(actor_var, actor_target_var):
            target_init_ops.append(tf.assign(target_var,var))
            soft_update_ops.append(tf.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
        for var, target_var in zip(critic_var, critic_target_var):
            target_init_ops.append(tf.assign(target_var,var))
            soft_update_ops.append(tf.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
        self.soft_update_ops = soft_update_ops
        self.sess.run(target_init_ops)
            
