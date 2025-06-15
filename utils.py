import gymnasium as gym
from network import ReccurentNetwork
import numpy as np
import matplotlib.pyplot as plt
import os


def run_with_saved_weights(filename, render=True):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    network = ReccurentNetwork(env.observation_space.shape[0], 
                          env.action_space.n, 
                          hidden_units=10)
    
    # Загружаем веса
    network.load_weights(filename)
    
    # Запускаем симуляцию
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = np.argmax(network.predict(state))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"Total reward: {total_reward}")
    env.close()

def plot_training_progress(avg_list, best_list):
    generations = list(range(1, len(avg_list) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_list, label='Средняя награда', linewidth=2)
    plt.plot(generations, best_list, label='Лучшая награда', linewidth=2)
    plt.xlabel('Поколение')
    plt.ylabel('Награда')
    plt.title('Прогресс обучения')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.show()


