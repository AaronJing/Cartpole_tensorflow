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
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  1 # discount factor
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS =  200# decay period
#hidden units
hidden_units = 24
#experience replay buffer size
memSize = 10000
#mini batch size
BATCH_SIZE = 100
#period we update our target network using estimate network
updateTN_period = 150

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
        #build memory list
        self.mem = []
    def add(self,state, action, reward, next_state, done):
        # experience requirements: state action reward next_state done
        experience = (state, action, reward,next_state,done)
        # if size of memory is bigger than the size we set pop the oldest one from list
        if len(self.mem) > memSize:
            del self.mem[0]
        # append experience into list
        self.mem.append(experience)
    def replayeable(self):
        return (len(self.mem) > BATCH_SIZE)
    def sample(self):
        # get random indexs in the range of memory size
        choice = np.random.choice(np.arange(len(self.mem)), size = BATCH_SIZE, replace = False)
        return [self.mem[i] for i in choice]

class network():
    def __init__(self, name='DQN'):
        with tf.variable_scope(name):
            # keep the scale of all layer roughly same
            self.w1 = tf.get_variable("w1",shape = [STATE_DIM,hidden_units],initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable("b1",shape = [1,hidden_units], initializer = tf.constant_initializer(0.0))
            self.w2 = tf.get_variable("w2",shape = [hidden_units,ACTION_DIM],initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable("b2",shape = [1,ACTION_DIM], initializer = tf.constant_initializer(0.0))

            self.logits1 = tf.matmul(state_in,self.w1) + self.b1
            self.output1 = tf.tanh(self.logits1)

            self.logits2 = tf.matmul(self.output1,self.w2) + self.b2
    def getQ(self):
        #no normalization here
        return self.logits2
"""
def TargetNetwork():
    w3 = tf.get_variable("w1",shape = [STATE_DIM,hidden_units],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b1",shape = [1,hidden_units], initializer = tf.constant_initializer(0.0))
    w4 = tf.get_variable("w2",shape = [hidden_units,ACTION_DIM],initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b2",shape = [1,ACTION_DIM], initializer = tf.constant_initializer(0.0))

    logits3 = tf.matmul(state_in,w3) + b3
    output3 = tf.tanh(logits3)

    logits2 = tf.matmul(output1,w2) + b2
    
    return logits2
# TODO: Define Network Graph
# double layer Q approximator, double layer
def DQNetwork():
    w1 = tf.get_variable("w1",shape = [STATE_DIM,hidden_units],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1",shape = [1,hidden_units], initializer = tf.constant_initializer(0.0))
    w2 = tf.get_variable("w2",shape = [hidden_units,ACTION_DIM],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2",shape = [1,ACTION_DIM], initializer = tf.constant_initializer(0.0))

    logits1 = tf.matmul(state_in,w1) + b1
    output1 = tf.tanh(logits1)

    logits2 = tf.matmul(output1,w2) + b2
    
    return logits2
"""
#build target network
TN = network(name = "TN")
#build estimate online learning network
DQN = network(name = "DQN")
#estimate value
q_values = DQN.getQ()
#estimate action
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

#target value
TN_q_values = TN.getQ()
#loss between estimate action and target
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)
#build memory
mem = experienceMem()
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


t = 0
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon- FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        # update experience mem
        mem.add(state,action,reward,next_state,done)   
        if mem.replayeable():
            batch = mem.sample()
            state_batch = [i[0] for i in batch]
            action_batch = [i[1] for i in batch]
            reward_batch = [i[2] for i in batch]
            next_state_batch = [i[3] for i in batch]
            target_batch = []
            #separate target network strategy
            TN_values_batch = session.run(TN.getQ(),feed_dict ={state_in: next_state_batch})
            for i in range(0,BATCH_SIZE):
                batch_done = batch[i][4]
                if batch_done :
                    target_batch.append(reward_batch[i])
                else:
                    target_batch.append(reward_batch[i] + GAMMA * np.max(TN_values_batch[i]))
            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })
        # after some period we update out target network     
        if t >= updateTN_period:
            # get parameter from estimate network
            DQN_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
            # get target network parameter 
            TN_parameter  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TN")
            parameter = []
            # update target network from parameter
            for DQN_parameter,TN_parameter in zip(DQN_parameter,TN_parameter):
                parameter.append(TN_parameter.assign(DQN_parameter))
            session.run(parameter)
            t = 0
        else:
            t = t + 1
        # Update
        state = next_state
        if done:
            break

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
