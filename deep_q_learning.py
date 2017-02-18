import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, SGD

import random

ENV_NAME = 'LunarLander-v2'

env = gym.make(ENV_NAME)
# To get repeatable results.
sd = 16
# np.random.seed(sd)
random.seed(sd)
env.seed(sd)
nb_actions = env.action_space.n

# env = wrappers.Monitor(env, '/tmp/experiment-1', force=True)

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error',  optimizer=Adam(lr=0.002,
              decay=2.25e-05))
print(model.summary())


def forward_pass(state):
    input = np.empty([1, 1, 8])
    input[0][0] = state
    return model.predict(input)[0]


def get_best_action(state):
    """Returns the index of the action with the highest Q-value, i.e.
        argMax(Q(nxt_state, all_actions))
    """
    state_q_values = forward_pass(state)
    return np.argmax(state_q_values)


def get_targets(state, action, reward, next_state):
    """
    Returns a set of target Q-values for a particular <s, a, r, s'> tuple

    """
    current_state_q_values = forward_pass(state)
    next_state_q_values = forward_pass(next_state)
    max_q_next_state = np.max(next_state_q_values)
    targets = np.empty([1, nb_actions])

    for i in range(nb_actions):
        if i == action:
            targets[0][i] = reward + (gamma * max_q_next_state)
        else:
            targets[0][i] = current_state_q_values[i]
    return targets


def choose_action(state, epsilon):
    r = np.random.uniform()
    if r < epsilon:
        action = np.floor(np.random.randint(nb_actions))
    else:
        action = get_best_action(state)
    return int(action)


class Memory(object):
    def __init__(self, memory_size=10000, experience_size=1):
        self.experiences = np.empty([0, experience_size], dtype=object)
        self.max_memory_size = memory_size

    def add_experience(self, experience):
        self.experiences = np.insert(self.experiences, 0,
                                     experience, axis=0)
        if len(self.experiences) > self.max_memory_size:
            self.experiences = np.delete(self.experiences,
                                         self.max_memory_size, axis=0)

    def sample_experiences(self, mini_batch_size):
        if(mini_batch_size > len(self.experiences)):
            rep_needed = True
        else:
            rep_needed = False
        s = self.experiences[np.random.choice(
                self.experiences.shape[0],
                mini_batch_size, replace=rep_needed)]
        return s


def pack_experience(state, action, reward, new_state):
    experience = np.empty([0])
    experience = np.append(experience, state)
    experience = np.append(experience, [action])
    experience = np.append(experience, [reward])
    experience = np.append(experience, new_state)
    return experience


def unpack_experience(experience):
    state = experience[0:8]
    action = experience[8]
    reward = experience[9]
    new_state = experience[10:18]
    return state, action, reward, new_state


def learn_from_replay_memories(memory, batch_size):
    sample_batch = memory.sample_experiences(batch_size)
    for e in sample_batch:
        state, action, reward, new_state = unpack_experience(e)
        targets = get_targets(state, action, reward, new_state)
        x = np.empty([1, 1, 8])
        x[0][0] = state
        model.train_on_batch(x, targets)

mini_batch_size = 5
replay_memory_size = 100
gamma = 0.9
epsilon = 0.30
max_steps_per_epoch = 1000
max_epochs = 1000

memory = Memory(replay_memory_size, 18)
total_reward = np.zeros(max_epochs)


for epoch in xrange(max_epochs):
    print "Episode #%i" % epoch
    state = env.reset()
    # env.render()
    current_step = 0
    epoch_done = False
    while current_step < max_steps_per_epoch and not epoch_done:
        action = choose_action(state, epsilon)
        new_state, reward, epoch_done, info = env.step(action)

        total_reward[epoch] = total_reward[epoch] + reward

        experience = pack_experience(state, action, reward, new_state)

        memory.add_experience(experience)
        current_step = current_step + 1
        state = new_state

        learn_from_replay_memories(memory, mini_batch_size)

    print "episode reward = %0.2f" % total_reward[epoch]
    if(not epoch % 10):
        print "last 10 episode avg = %0.2f" % np.average(
            total_reward[epoch-10:epoch])

    # gamma = gamma * gamma
    # print "memory size = %i" % len(memory.experiences)
