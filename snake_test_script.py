# # from game_environment_parallel import Snake
# from game_environment import Snake
# import numpy as np


# # env = Snake(board_size=10, frames=2, n_games=3)
# env = Snake(board_size=10, frames=2)
# s = env.reset()
# env.print_game()
# '''
# done = 0
# while(not done):
#     # action = np.random.choice([-1, 0, 1], 1)[0]
#     # instead of random action, take input from user
#     action = int(input('Enter action [-1, 0, 1] : '))
#     # print(action)
#     s, r, done, info = env.step(action)
#     # print(env._snake_direction)
#     # for i, x in enumerate(env._snake):
#         # print(i, x.row, x.col)
#     env.print_game()
# '''

import torch
from game_environment import Snake
import numpy as np

# Last inn den trenede modellen
from agent import DeepQLearningAgent  # Importer agentklassen



# agent = DeepQLearningAgent(board_size=10, frames=2, n_actions=3)
agent = DeepQLearningAgent(board_size=10, frames=2, n_actions=3)

agent.load_model(file_path="C:\\Users\\beyan\\Desktop\\VSC\\NN\\snake-rl\\models\\v17.1", iteration=200000)


# Initialiser miljøet
env = Snake(board_size=10, frames=2)
s = env.reset()

done = False
total_reward = 0  # For å spore poengsum

while not done:
    # Bruk modellen til å velge handling
    s_normalized = torch.from_numpy(s).float().unsqueeze(0).to('cuda')  # Normaliser og send til GPU
    with torch.no_grad():
        q_values = agent._model(s_normalized)
    action = torch.argmax(q_values).item()  # Velg handling med høyest Q-verdi

    # Utfør handlingen i miljøet
    s, reward, done, info = env.step(action)
    total_reward += reward

    # Vis spilltilstanden
    env.print_game()

print(f"Spillet er over! Total poengsum: {total_reward}")
