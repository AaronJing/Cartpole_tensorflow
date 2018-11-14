import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 10  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.999 # discount factor
INITIAL_EPSILON = 0.8 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
hidden_units = 32
memSize = 10000
BATCH_SIZE = 128

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# experience replay
class experienceMem():
    def __init__(self):
        self.mem =[]
    def add(self,state, action, reward, next_state,done):
        experience = (state, action, reward,next_state,done)
        if len(self.mem) >= memSize :
            del self.mem[0]
        self.mem.append(experience)
    def replayeable(self):
        return (len(self.mem) > 200)
    def sample(self):
        choice = np.random.choice(np.arange(len(self.mem)), size = BATCH_SIZE, replace = False)
        return [self.mem[i] for i in choice]
"""
class network():
    def __init__(self, name='DQN'):
        with tf.variable_scope(name):
            self.w1 = tf.get_variable("w1",shape = [STATE_DIM,hidden_units],initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable("b1",shape = [1,hidden_units], initializer = tf.constant_initializer(0.0))
            self.w2 = tf.get_variable("w2",shape = [hidden_units,hidden_units],initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable("b2",shape = [1,hidden_units], initializer = tf.constant_initializer(0.0))
            self.w3 = tf.get_variable("w3",shape = [hidden_units,ACTION_DIM],initializer=tf.contrib.layers.xavier_initializer())
            self.b3 = tf.get_variable("b3",shape = [1,ACTION_DIM], initializer = tf.constant_initializer(0.0))
            self.logits1 = tf.matmul(state_in,self.w1) + self.b1
            self.output1 = tf.nn.relu(self.logits1)
            self.logits2 = tf.matmul(self.output1,self.w2) + self.b2
            self.output2 = tf.nn.relu(self.logits2)
            self.logits3 = tf.matmul(self.output2,self.w3) + self.b3
    def getQ(self):
        return self.logits3
"""
class network():
    def __init__(self, name):
        with tf.variable_scope(name):
            self.w1 = tf.Variable(tf.truncated_normal([STATE_DIM,hidden_units]))
            self.b1 = tf.Variable(tf.constant(0.1,shape=[hidden_units]))
            self.w2 = tf.Variable(tf.truncated_normal([hidden_units,hidden_units]))
            self.b2 = tf.Variable(tf.constant(0.1,shape=[hidden_units]))
            self.w3 = tf.Variable(tf.truncated_normal([hidden_units,ACTION_DIM]))
            self.b3 = tf.Variable(tf.constant(0.1,shape=[ACTION_DIM]))
            self.logits1 = tf.matmul(state_in,self.w1) + self.b1
            self.output1 = tf.nn.relu(self.logits1)
            self.logits2 = tf.matmul(self.output1,self.w2) + self.b2
            self.output2 = tf.nn.relu(self.logits2)
            self.logits3 = tf.matmul(self.output2,self.w3) + self.b3
    def getQ(self):
        return self.logits3       

# TODO: Network outputs
TN = network(name = "TN")
DQN = network(name = "DQN")
q_values = TN.getQ()

# TODO: Loss/Optimizer Definition
DQN_q_values = DQN.getQ()
DQN_q_action = tf.reduce_sum(tf.multiply(DQN_q_values, action_in), reduction_indices=1)
loss = tf.reduce_sum(tf.square(target_in - DQN_q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)


def updateTN():
    DQN_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
    TN_parameter  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TN")
    parameter = []
    for DQN_parameter,TN_parameter in zip(DQN_parameter,TN_parameter):
        parameter.append(TN_parameter.assign(DQN_parameter))
    return parameter

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

mem = experienceMem()
t = 0
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
   
    epsilon = epsilon - epsilon  / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        t+=1
         #session.run(TN.getQ(),feed_dict={state_in:next_state})
       
        # q_TN = session.run(TN.getQ(),feed_dict = {state_in:[next_state]})
        # update experience mem
        """
        if done and t < 500:
            mem.add(state,action,-100,next_state,nextstate_q_values,done)
        else:
        """
        mem.add(state,action,reward,next_state,done)
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        if mem.replayeable():
            # do train step
            batch = mem.sample()
            state_batch = [i[0] for i in batch]
            action_batch = [i[1] for i in batch]
            reward_batch = [i[2] for i in batch]
            next_state_batch = [i[3] for i in batch]
            target_batch = []
            #q_values_batch = q_values.eval(feed_dict ={state_in: next_state_batch})
            DQN_values_batch = DQN_q_values.eval(feed_dict={state_in:next_state_batch})
            TN_values_batch = session.run(TN.getQ(),feed_dict={state_in:next_state_batch})
            for i in range(0,BATCH_SIZE):
                batch_done = batch[i][4]
                dqn_action = np.argmax(DQN_values_batch[i])
                if not batch_done :
                    target_batch.append(reward_batch[i] + GAMMA * TN_values_batch[i][dqn_action])
                else:
                    target_batch.append(reward_batch[i]) 
            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })
        """
        else:
            # do one-step training
            if not done:
                target = reward+ GAMMA * np.max(nextstate_q_values)
            else:
                target = reward

            # Do onetraining step
            session.run([optimizer], feed_dict={
                target_in: [target],
                action_in: [action],
                state_in: [state]
            })
        """     
        # Update
        
        
        if done:
            session.run(updateTN())             
            break
        state = next_state
    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                #env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
