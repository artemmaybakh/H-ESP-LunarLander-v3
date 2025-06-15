import gymnasium as gym
from model import HESP

# Запуск алгоритма
if __name__ == "__main__":
        env = gym.make("LunarLander-v3")
        hesp = HESP(
            env,
            population_size=50,  # Популяция
            hidden_units=10,     # Количество нейронов на скрытом слое
            L1_size=20,          # Количество входных нейронов
            L2_size=300,         # Количество выходных нейронов
        )
        avg, best = hesp.evolve(generations=919)
        env.close()