from ple import PLE
from ple.games.flappybird import FlappyBird
import random
import tensorflow as tf
from collections import deque
import numpy as np

game = FlappyBird()
env = PLE(game, display_screen=False, add_noop_action=True)
ACTIONS = env.getActionSet()
env.init()
state = list(env.getGameState().values())

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, activation="relu", input_shape=np.array(state).shape))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(2, activation="linear"))
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
model.summary()

MEM_SIZE = 10_000
BATCH_SIZE = 32
EPISODES = 5_000
DISCOUNT = 0.95
MAX_EPSILON = 1
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.00005

memory = deque(maxlen=MEM_SIZE)
epsilon = MAX_EPSILON
total_steps = 0
max_reward = 0


def init_replay_memory():
    while len(memory) <= BATCH_SIZE:
        print(len(memory))
        env.reset_game()
        done = False
        # state, reward, done, info = env.step(env.action_space.sample())
        env.reset_game()
        state = list(env.getGameState().values())

        while not env.game_over() and len(memory) <= BATCH_SIZE:
            action = np.argmax(random.choice(ACTIONS))
            reward = env.act(ACTIONS[action])
            new_state = list(env.getGameState().values())

            if env.game_over():
                new_state = np.zeros(state.shape)

            memory.append((state, action, reward, new_state))
            state = new_state

# for i in range(0, 10_000):
#     if env.game_over():
#         break
#
#     out = model.predict(np.array([state]))
#     action = np.argmax(out)
#     reward = env.act(ACTIONS[action])
#     new_state = list(env.getGameState().values())
#
#     memory.append((state, action, reward, new_state))
#     state = new_state


def choose_action():
    if random.random() < epsilon:
        return int(random.random() * len(ACTIONS))
    return np.argmax(model.predict(np.array([state])))


def update_network():
    batch = random.choices(memory, k=BATCH_SIZE)
    x = np.zeros((BATCH_SIZE, 8))
    y = np.zeros((BATCH_SIZE, 2))
    for i, mem in enumerate(batch):
        state, action, reward, new_state = mem
        x[i] = state

        update = reward
        if not (new_state == [0] * len(new_state)):
            update = reward + DISCOUNT * np.max(model.predict(np.array([new_state])))

        q_vals = model.predict(np.array([state]))
        q_vals[action] = update
        y[i] = q_vals

        # model.fit(state, q_vals, verbose=0)
    # model.fit(x, y, batch_size=32, epochs=1, verbose=0)
    model.fit(x, y, epochs=1, verbose=0)


init_replay_memory()

for episode in range(0, EPISODES):
    env.reset_game()
    state = list(env.getGameState().values())
    ep_reward = 0

    while not env.game_over():
        total_steps += 1

        if epsilon > MIN_EPSILON:
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-EPSILON_DECAY * total_steps)

        action = choose_action()
        reward = env.act(ACTIONS[action])
        new_state = list(env.getGameState().values())
        ep_reward += reward

        if env.game_over():
            new_state = [0] * len(new_state)
            print("Episode {} complete. Reward: {} Epsilon: {} Total steps: {}".format(
                episode, ep_reward, epsilon, total_steps))
            if ep_reward > max_reward:
                max_reward = ep_reward
                model.save("Models/{}.h5".format(max_reward))

        memory.append((state, action, reward, new_state))
        state = new_state
