import gymnasium as gym
from model import HESP
from utils import run_with_saved_weights, plot_training_progress

MODE = 'train'

if __name__ == "__main__":
    if MODE == 'train':
        env = gym.make("LunarLander-v3")
        hesp = HESP(
            env,
            population_size=50,  # Популяция
            hidden_units=10,     # Количество нейронов на скрытом слое
            L1_size=20,          # Количество входных нейронов
            L2_size=300,         # Количество выходных нейронов
        )
        avg, best = hesp.evolve(generations=10000)
        hesp.save_best_network("best_network_weights.json")
        env.close()
        plot_training_progress(avg, best)
    else:
        run_with_saved_weights("best_network_weights1.json")