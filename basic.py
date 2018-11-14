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
GAMMA = 0.999 # discount factor
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON =  0.01# final value of epsil
EPSILON_DECAY_STEPS = 200 # decay period
hidden_units = 32
BATCH_SIZE = 128
MEM_SIZE = 10000
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

class experienceMem():
    def __init__(self):
        self.mem = []
    def add(self,state, action, next_state, reward, done):
        experience = (state, action, next_state, reward,done)
        if len(self.mem) > MEM_SIZE:
            del self.mem[0]
        self.mem.append(experience)
    def replayeable(self):
        return (len(self.mem) > BATCH_SIZE)
    def sample(self):
        choice = np.random.choice(np.arange(len(self.mem)), size = BATCH_SIZE, replace = False)
        return [self.mem[i] for i in choice]

# TODO: Define Network Graph
class network():
    def __init__(self, name='DQN'):
        with tf.variable_scope(name):
            self.w1 = tf.get_variable("w1",shape = [STATE_DIM,hidden_units],initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable("b1",shape = [1,hidden_units], initializer = tf.constant_initializer(0.0))
            self.w2 = tf.get_variable("w2",shape = [hidden_units,ACTION_DIM],initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable("b2",shape = [1,ACTION_DIM], initializer = tf.constant_initializer(0.0))

            self.logits1 = tf.matmul(state_in,self.w1) + self.b1
            self.output1 = tf.tanh(self.logits1)

            self.logits2 = tf.matmul(self.output1,self.w2) + self.b2
    def getQ(self):
        return self.logits2

basic = network()
# TODO: Network outputs
q_values = basic.getQ()
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)

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
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        mem.add(state,action,next_state,reward,done)

        if mem.replayeable():
            experience_mb = mem.sample()
            state_mb = [i[0] for i in experience_mb]
            action_mb = [i[1] for i in experience_mb]
            next_state_mb = [i[2] for i in experience_mb]
            reward_mb = [i[3] for i in experience_mb]

            nextstate_q_values_mb = q_values.eval(feed_dict = {state_in: next_state_mb})
            target_mb = []
            for i in range(0,BATCH_SIZE):
                if experience_mb[i][4]:
                   target_mb.append(reward_mb[i]) 
                else:
                   target_mb.append(reward_mb[i]+ np.max(nextstate_q_values_mb[i]))
            session.run([optimizer], feed_dict={
                target_in: target_mb,
                action_in: action_mb,
                state_in: state_mb
            })
        """
        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        if done: 
            target = reward
        else:
            target = reward + GAMMA * np.max(nextstate_q_values)

        # Do one training step
        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
        })
        """
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
                # env.render()
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
