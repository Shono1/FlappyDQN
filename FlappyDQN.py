from ple import PLE
from ple.games.flappybird import FlappyBird
import random
import tensorflow as tf
from collections import deque

game = FlappyBird()

print(game.getActions())
env = PLE(game, display_screen=True, add_noop_action=True)
print(env.getActionSet())
env.init()
state = list(env.getGameState().values())

model = tf.keras.Sequential()
model.layers.append(tf.keras.layers.InputLayer((8, 1), batch_size=32))
model.layers.append(tf.keras.layers.Dense(6, activation="relu"))
model.layers.append(tf.keras.layers.Dense(6, activation="relu"))
model.layers.append(tf.keras.layers.Dense(2, activation="relu"))
model.compile()

memory = deque(maxlen=100_000)


for i in range(0, 1000):
    if env.game_over():
        break

    action = random.choice(env.getActionSet())
    reward = env.act(action)
    new_state = list(env.getGameState().values())

    memory.append((state, action, reward, new_state))
    state = new_state

for mem in memory:
    print(mem)