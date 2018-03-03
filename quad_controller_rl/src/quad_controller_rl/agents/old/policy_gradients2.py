"""Policy gradient agent."""
import os
import random
from collections import namedtuple
from collections import deque

import numpy as np
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent

from keras.layers import Input, Dense, Lambda, LeakyReLU, Conv1D, Add, Multiply
from keras.layers import Activation, Concatenate, BatchNormalization, Flatten, Reshape
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import regularizers
import tensorflow as tf

class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=size)  # internal memory (list)
        self.Exp = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(self.Exp(state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    #0.15 0.3
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        
        net = Dense(100)(states)
        net = LeakyReLU(alpha=0.2)(net)
        net = BatchNormalization()(net)
        #net = Activation('sigmoid')(net)
        #net = LeakyReLU(alpha=0.25)(net)
        net = Dense(150,kernel_initializer='VarianceScaling',activation='selu')(net)
        #net = Activation('sigmoid')(net)
        #net = LeakyReLU(alpha=0.25)(net)
        net = Dense(50,kernel_initializer='VarianceScaling',activation='selu')(net)
        #net = Activation('sigmoid')(net)
        #net = LeakyReLU(alpha=0.3)(net)
        
        raw_actions = Dense(units=self.action_size, activation='sigmoid',name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = Input(shape=(self.action_size,))
        
        #loss = K.mean(-action_gradients * actions)
        # Incorporate any additional losses here (e.g. from regularizers)
        params_grad = tf.gradients(raw_actions, self.model.trainable_weights, 
                                   grad_ys=-action_gradients,colocate_gradients_with_ops=True)
        grads = zip(params_grad, self.model.trainable_weights)
        
        # Define optimizer and training function
        #optimizer = optimizers.Adam()
        #updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        updates_op = tf.train.AdamOptimizer(0.001).apply_gradients(grads)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=[updates_op])
        
        
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.build_model()
        
    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = Input(shape=(self.state_size,), name='states')
        actions = Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = Dense(units=200)(states)
        net_states = LeakyReLU(alpha=0.2)(net_states)
        net_states = BatchNormalization()(net_states)
        net_states = Dense(units=150,kernel_initializer='VarianceScaling',activation='selu')(net_states)
        #net_states = LeakyReLU(alpha=0.2)(net_states)
        #net_states = BatchNormalization()(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = Dense(units=30)(actions)
        net_actions = LeakyReLU(alpha=0.2)(net_actions)
        net_actions = BatchNormalization()(net_actions)
        net_actions = Dense(units=70,kernel_initializer='VarianceScaling',activation='selu')(net_actions)
        #net_actions = LeakyReLU(alpha=0.2)(net_actions)
        #net_actions = BatchNormalization()(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = Concatenate(axis=-1)([net_states, net_actions])
        net = Dense(units=100,kernel_initializer='VarianceScaling',activation='selu')(net)
        #net = LeakyReLU(alpha=0.2)(net)
        #net = BatchNormalization()(net)
        net = Dense(units=50)(net)
        net = LeakyReLU(alpha=0.2)(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

class DDPG2(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        # Save episode stats
        #self.stats_filename = os.path.join(
        #    util.get_param('out'),
        #    "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        #self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.episode_num = 1
        #print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]
        
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        #self.state_size = np.prod(self.task.observation_space.shape)
        self.state_size = 9
        #self.action_size = np.prod(self.task.action_space.shape)
        self.action_size = 1
        
        # Actor (Policy) Model
        #self.action_low = self.task.action_space.low
        #self.action_high = self.task.action_space.high
        self.action_low = self.task.action_space.low[2]
        self.action_high = self.task.action_space.high[2]
        
        #self.action_high[-3:] = 0.0
        
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 100
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.97  # discount factor
        self.tau = 0.0001  # for soft update of target parameters
        
        #
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        #self.replay_start_size = 150
        
    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        
    def step(self, state, reward, done):
        # Choose an action
        print('r:{:.3}'.format(reward),end="\r")
        action = self.act(state)
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            
        # Learn, if enough samples are available in memory
        #if (len(self.memory) > self.batch_size) and (len(self.memory) > self.replay_start_size):
        if (len(self.memory) > self.batch_size):
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
        self.last_state = state
        self.last_action = action
        
        if done:
            # Write episode stats
            #self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            print('total reward={:7.4f}'.format(self.total_reward))
            self.reset_episode_vars()
        
        #only z force
        a = np.zeros((action.shape[0],6))
        a[:,2] = action
        return a

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        noise = self.noise.sample()
        return actions + noise # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + (self.gamma * Q_targets_next)* (1. - dones)
        
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function
        
        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)