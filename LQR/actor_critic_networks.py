import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
from tflearn.layers.conv import residual_block

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 200
MAX_EPISODES_TEST = 50
# Max episode length
MAX_EP_STEPS = 100
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 1e-7
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 1e-7
# Discount factor
GAMMA = 0.9
# Soft target update param
TAU = 0.0001

critic_hidden_layer1_size = 128
critic_hidden_layer2_size = 64

actor_hidden_layer1_size = 256
actor_hidden_layer2_size = 128
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 20000
SMALL_BUFFER_SIZE = 2000
MINIBATCH_SIZE = 64
# Update Times per step
R = 1

REWARD_SPACE_DIM = 2

# ===========================
#   Actor and Critic DNNs
# ===========================

def replace_none_with_zero(l):
    return [0.0 if i==None else i for i in l]

def ResDense(x, input_size, bottle_size):
    w_init_out1 = tflearn.initializations.xavier(seed=RANDOM_SEED)
    x1 = tflearn.fully_connected(x, bottle_size, activation='relu', weights_init=w_init_out1)
    w_init_out2 = tflearn.initializations.xavier(seed=RANDOM_SEED)
    x2 = tflearn.fully_connected(x1, input_size, activation='relu', weights_init=w_init_out2)
    return x + x2
    
class ActorNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        
        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
        self.mu_gradients = tf.gradients(self.scaled_out, self.network_params)
         
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)
        
        self.get_grad_global_norm = tf.global_norm(replace_none_with_zero(self.actor_gradients))
        #self.names_layers = [v.name for v in self.network_params]
        self.get_grad_norm = tf.global_norm(replace_none_with_zero(self.mu_gradients))

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        bn_inputs = tflearn.batch_normalization(inputs, trainable=True)
        w_init_layer1 = tflearn.initializations.xavier(seed=RANDOM_SEED)
        layer1 = tflearn.fully_connected(bn_inputs, actor_hidden_layer1_size, activation='linear', weights_init=w_init_layer1)
        layer2 = ResDense(layer1, actor_hidden_layer1_size, actor_hidden_layer2_size)
        w_init_out = tflearn.initializations.uniform(minval=-3e-9, maxval=3e-9, seed=RANDOM_SEED)
        out = tflearn.fully_connected(layer2, self.a_dim, weights_init=w_init_out)
        out = tflearn.activation(out, 'tanh')
        scaled_out = tf.multiply(out, self.action_bound)      
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        return self.sess.run([self.optimize, self.get_grad_global_norm, self.get_grad_norm], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def get_actor_gradients(self, inputs, a_gradient):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
    

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        self.gradients_weights = tf.placeholder(tf.float32, [REWARD_SPACE_DIM, 1], name="GRADIENTS_WEIGHTS")
        
        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, REWARD_SPACE_DIM])
        
        regul = tf.reduce_sum([tf.nn.l2_loss(weights) for weights in self.network_params]) * 0.04
        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out) + regul
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)
    
        self.action_grads = tf.gradients(tf.matmul(self.out, self.gradients_weights), self.action)
        self.get_grad_global_norm = tf.global_norm(replace_none_with_zero(self.action_grads))

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        bn_inputs = tflearn.batch_normalization(inputs, trainable=True)
        action = tflearn.input_data(shape=[None, self.a_dim])
        bn_action = tflearn.batch_normalization(action, trainable=True)
        concat = lambda x:tf.concat([x[0],x[1]],axis=1)
        
        w_state_layer = tflearn.initializations.xavier(seed=RANDOM_SEED)
        state_layer = tflearn.fully_connected(bn_inputs, critic_hidden_layer1_size, activation='linear', weights_init=w_state_layer)
        concat_layer = concat([state_layer, bn_action])
        concat_layer = ResDense(concat_layer, (critic_hidden_layer1_size + self.a_dim), int((critic_hidden_layer1_size + self.a_dim) / 2))
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init_out = tflearn.initializations.uniform(minval=-3e-9, maxval=3e-9, seed=RANDOM_SEED)
        out = tflearn.fully_connected(concat_layer, REWARD_SPACE_DIM, weights_init=w_init_out)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions, grad_weights):
        return self.sess.run([self.action_grads, self.get_grad_global_norm], feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.gradients_weights: grad_weights
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)