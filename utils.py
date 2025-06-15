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

def visualize_network(network, gen, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    input_size = len(network.hidden_neurons[0].input_weights)
    hidden_size = len(network.hidden_neurons)
    output_size = len(network.output_neurons)

    layer_x = [0, 1, 2]  # input -> hidden -> output

    def centered_y(size):
        spacing = 1.0
        return np.linspace(-spacing * (size - 1) / 2, spacing * (size - 1) / 2, size)

    input_y = centered_y(input_size)
    hidden_y = centered_y(hidden_size)
    output_y = centered_y(output_size)

    # INPUT neurons
    for i, y in enumerate(input_y):
        ax.plot(layer_x[0], y, 'o', color='gray', markersize=14, zorder=3)

    # HIDDEN neurons
    for j, hn in enumerate(network.hidden_neurons):
        x_h, y_h = layer_x[1], hidden_y[j]
        ax.plot(x_h, y_h, 'o', color='blue', markersize=14, zorder=3)

        # Connections from input
        for i, w in enumerate(hn.input_weights):
            y_i = input_y[i]
            linewidth = max(0.5, min(abs(w) * 2, 4))
            color = 'green' if w > 0 else 'red'
            ax.plot([layer_x[0], x_h], [y_i, y_h], linewidth=linewidth, color=color, alpha=0.6, zorder=1)

        # Recurrent self-connection
        rw = hn.recurrent_weight
        if abs(rw) > 1e-3:
            circle_radius = 0.2
            circle = plt.Circle((x_h, y_h), circle_radius,
                                fill=False,
                                color='green' if rw > 0 else 'red',
                                linewidth=max(0.5, min(abs(rw) * 2, 4)),
                                alpha=0.6,
                                linestyle='--')
            ax.add_patch(circle)

    # OUTPUT neurons
    for k, on in enumerate(network.output_neurons):
        y_o = output_y[k]
        ax.plot(layer_x[2], y_o, 'o', color='orange', markersize=14, zorder=3)
        for j, w in enumerate(on.weights):
            y_h = hidden_y[j]
            linewidth = max(0.5, min(abs(w) * 2, 4))
            color = 'green' if w > 0 else 'red'
            ax.plot([layer_x[1], layer_x[2]], [y_h, y_o], linewidth=linewidth, color=color, alpha=0.6, zorder=1)

    # Легенда
    ax.plot([], [], color='green', label='положительный вес')
    ax.plot([], [], color='red', label='отрицательный вес')
    ax.plot([], [], color='black', linestyle='--', label='рекуррентная связь (петля)')
    ax.legend(loc='upper right')

    # Сохранение
    if filename is None:
        os.makedirs("topologies", exist_ok=True)
        filename = f"topologies/topology_gen_{gen}.png"

    plt.title(f"Топология на поколении {gen}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


