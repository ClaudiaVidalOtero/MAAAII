import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import os

# ===========================================================
# CONFIGURACIÓN
# ===========================================================

ENV_NAME = 'Pendulum-v1'
RENDER = False
PERTURBACIONES = False
PERTURB_PROB = 0.05
PERTURB_VALS = [-2.0, 2.0]

N_ANGLE_BINS = 30 #
N_VELOCITY_BINS = 30 #
N_ACTIONS = 17

ANGLE_MIN, ANGLE_MAX = -np.pi, np.pi
VEL_MIN, VEL_MAX = -8.0, 8.0
ACTION_MIN, ACTION_MAX = -2.0, 2.0

NUM_EPISODES = 10000
MAX_STEPS = 200
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995 #
ALPHA = 0.05 # 

angle_bins = np.linspace(ANGLE_MIN, ANGLE_MAX, N_ANGLE_BINS + 1)
vel_bins = np.linspace(VEL_MIN, VEL_MAX, N_VELOCITY_BINS + 1)
action_list = np.linspace(ACTION_MIN, ACTION_MAX, N_ACTIONS)

# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================

def discretize_state(obs):
    cos_theta, sin_theta, theta_dot = obs
    theta = np.arctan2(sin_theta, cos_theta)
    angle_idx = np.digitize(theta, angle_bins) - 1
    vel_idx = np.digitize(theta_dot, vel_bins) - 1
    angle_idx = np.clip(angle_idx, 0, N_ANGLE_BINS - 1)
    vel_idx = np.clip(vel_idx, 0, N_VELOCITY_BINS - 1)
    return (angle_idx, vel_idx)

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(N_ACTIONS)
    else:
        return np.argmax(Q[state])

def apply_perturbation(action):
    if np.random.rand() < PERTURB_PROB:
        return [np.random.choice(PERTURB_VALS)]
    else:
        return [action]

def q_learning(env):
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    returns = []
    epsilon = EPSILON_START

    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        state = discretize_state(obs)
        total_reward = 0

        for t in range(MAX_STEPS):
            action_idx = epsilon_greedy(Q, state, epsilon)
            action = action_list[action_idx]
            if PERTURBACIONES:
                action = apply_perturbation(action)[0]

            next_obs, reward, terminated, truncated, _ = env.step([action])
            next_state = discretize_state(next_obs)
            done = terminated or truncated

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + GAMMA * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action_idx]
            Q[state][action_idx] += ALPHA * td_error

            total_reward += reward
            state = next_state

            if done:
                break

        returns.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if ep % 500 == 0:
            print(f"Episodio {ep}/{NUM_EPISODES}, retorno medio últimos 500: {np.mean(returns[-500:]):.2f}")

    return Q, returns

def visualizar_politica(policy_det, n_episodios=5):
    env_vis = gym.make(ENV_NAME, render_mode='human')

    for ep in range(n_episodios):
        obs, _ = env_vis.reset()
        state = discretize_state(obs)
        total_reward = 0
        print(f"\n[EPISODIO {ep+1}]")

        for t in range(MAX_STEPS):
            action_idx = policy_det.get(state, np.random.choice(N_ACTIONS))
            action = action_list[action_idx]

            obs, reward, terminated, truncated, _ = env_vis.step([action])
            total_reward += reward
            state = discretize_state(obs)

            env_vis.render()

            if terminated or truncated:
                break

        print(f"Retorno del episodio {ep+1}: {total_reward:.2f}")

    env_vis.close()


def evaluar_politica(policy_det, n_episodios=100):
    env_eval = gym.make(ENV_NAME, render_mode='human' if RENDER else None)
    retornos = []

    for _ in range(n_episodios):
        obs, _ = env_eval.reset()
        state = discretize_state(obs)
        total_reward = 0

        for _ in range(MAX_STEPS):
            action_idx = policy_det.get(state, np.random.choice(N_ACTIONS))
            action = action_list[action_idx]
            obs, reward, terminated, truncated, _ = env_eval.step([action])
            total_reward += reward
            state = discretize_state(obs)
            if terminated or truncated:
                break

        retornos.append(total_reward)

    env_eval.close()
    return np.mean(retornos), np.std(retornos)

# ===========================================================
# EJECUCIÓN
# ===========================================================

env = gym.make(ENV_NAME, render_mode=None)
Q, returns = q_learning(env)

policy_det = {s: np.argmax(a) for s, a in Q.items()}

os.makedirs('resultados', exist_ok=True)

# Guardar gráfico
plt.plot(returns)
plt.xlabel('Episodios')
plt.ylabel('Retorno')
plt.title('Q-Learning - Pendulum-v1')
plt.savefig('resultados/grafico_qlearning.png')
plt.close()

# Evaluar
media, std = evaluar_politica(policy_det)
print(f"\nEvaluación política final: Retorno medio = {media:.2f}, Std = {std:.2f}")

# Evaluar rendimiento promedio
media, std = evaluar_politica(policy_det)
print(f"\nEvaluación política final: Retorno medio = {media:.2f}, Std = {std:.2f}")

# Visualizar política entrenada
visualizar_politica(policy_det, n_episodios=5)
