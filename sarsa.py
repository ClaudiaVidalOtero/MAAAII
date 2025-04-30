import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict, namedtuple
import os

# ===========================================================
# CONFIGURACION DE HIPERPARAMETROS Y OPCIONES
# ===========================================================

# Entorno
ENV_NAME = 'Pendulum-v1'
RENDER = True  # True para  visualizar el entorno
PERTURBACIONES = False  # True para introducir perturbaciones
PERTURB_PROB = 0.05  # Probabilidad de perturbacion
PERTURB_VALS = [-2.0, 2.0]  # Valores de las perturbaciones

# Discretizacion
N_ANGLE_BINS = 40
N_VELOCITY_BINS = 40
N_ACTIONS = 11

# Rango de estados y acciones
ANGLE_MIN, ANGLE_MAX = -np.pi, np.pi
VEL_MIN, VEL_MAX = -8.0, 8.0
ACTION_MIN, ACTION_MAX = -2.0, 2.0

# Algoritmo SARSA
NUM_EPISODES = 10000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9
MAX_STEPS = 200  # ~20Hz x 10 segundos
ALPHA = 0.1  # Tasa de aprendizaje fija para SARSA

# ===========================================================
# DISCRETIZACION
# ===========================================================

angle_bins = np.linspace(ANGLE_MIN, ANGLE_MAX, N_ANGLE_BINS+1)
vel_bins = np.linspace(VEL_MIN, VEL_MAX, N_VELOCITY_BINS+1)
action_list = np.linspace(ACTION_MIN, ACTION_MAX, N_ACTIONS)

EpisodeStep = namedtuple('EpisodeStep', ['state', 'action', 'reward'])

# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================

def discretize_state(obs):
    cos_theta, sin_theta, theta_dot = obs
    theta = np.arctan2(sin_theta, cos_theta)
    angle_idx = np.digitize(theta, angle_bins) - 1
    vel_idx   = np.digitize(theta_dot, vel_bins) - 1
    # Hacemos un "clip" para asegurarnos de que los índices estén dentro de los límites
    angle_idx = np.clip(angle_idx, 0, N_ANGLE_BINS-1)
    vel_idx   = np.clip(vel_idx, 0, N_VELOCITY_BINS-1)
    return (angle_idx, vel_idx)

def create_initial_policy():
    return defaultdict(lambda: np.ones(N_ACTIONS)/N_ACTIONS)

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


# ===========================================================
# FUNCIONES PRINCIPALES
# ===========================================================

def generate_episode(env, policy, epsilon, perturbations=False):
    episode = []
    obs, _ = env.reset()
    state_idx = discretize_state(obs)

    for t in range(MAX_STEPS):
        a_idx = epsilon_greedy(Q, state_idx, epsilon)
        action = action_list[a_idx]
        if perturbations:
            action = apply_perturbation(action)[0]

        next_obs, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated

        episode.append(EpisodeStep(state_idx, a_idx, reward))

        state_idx = discretize_state(next_obs)
        if done:
            break
    return episode

def update_Q_and_policy(episode, Returns_count, Q, policy):
    G = 0.0
    visited = set()
    for step in reversed(episode):
        G = GAMMA * G + step.reward
        sa = (step.state, step.action)
        if sa not in visited:
            Returns_count[step.state][step.action] += 1
            alpha = 1 / Returns_count[step.state][step.action]
            Q[step.state][step.action] += alpha * (G - Q[step.state][step.action])

            # Actualizar politica e-soft
            best_actions = np.argwhere(Q[step.state] == np.max(Q[step.state])).flatten()
            for a in range(N_ACTIONS):
                if a in best_actions:
                    policy[step.state][a] = (1 - EPSILON_CURRENT) / len(best_actions) + EPSILON_CURRENT / N_ACTIONS
                else:
                    policy[step.state][a] = EPSILON_CURRENT / N_ACTIONS

            visited.add(sa)

# ===========================================================
# FUNCIONES PRINCIPALES - SARSA
# ===========================================================

def sarsa_control(env):
    global EPSILON_CURRENT
    returns = []
    EPSILON_CURRENT = EPSILON_START

    for ep in range(1, NUM_EPISODES+1):
        obs, _ = env.reset()
        state_idx = discretize_state(obs)

        # Elegimos la primera acción
        a_idx = epsilon_greedy(Q, state_idx, EPSILON_CURRENT)
        action = action_list[a_idx]
        if PERTURBACIONES:
            action = apply_perturbation(action)[0]

        total_reward = 0

        for t in range(MAX_STEPS):
            next_obs, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            total_reward += reward

            next_state_idx = discretize_state(next_obs)

            # Elegimos siguiente acción
            next_a_idx = epsilon_greedy(Q, next_state_idx, EPSILON_CURRENT)
            next_action = action_list[next_a_idx]
            if PERTURBACIONES:
                next_action = apply_perturbation(next_action)[0]

            # Actualización SARSA
            td_target = reward + GAMMA * Q[next_state_idx][next_a_idx]
            td_error = td_target - Q[state_idx][a_idx]
            Q[state_idx][a_idx] += ALPHA * td_error

            # Avanzamos estado y acción
            state_idx = next_state_idx
            a_idx = next_a_idx
            action = next_action

            if done:
                break

        returns.append(total_reward)

        # Decay epsilon
        EPSILON_CURRENT = max(EPSILON_END, EPSILON_CURRENT * EPSILON_DECAY)

        if ep % 500 == 0:
            print(f"Episodio {ep}/{NUM_EPISODES}, retorno medio últimos 500: {np.mean(returns[-500:]):.2f}")

    return returns

def evaluar_politica(policy_det, n_episodios=100):

    env_eval = gym.make(ENV_NAME, render_mode='human')
    retornos = []

    for ep in range(n_episodios):
        obs, _ = env_eval.reset()
        state_idx = discretize_state(obs)

        total_reward = 0
        for t in range(MAX_STEPS):
            action_idx = policy_det.get(state_idx, np.random.choice(N_ACTIONS))
            action = action_list[action_idx]
            obs, reward, terminated, truncated, _ = env_eval.step([action])
            total_reward += reward
            state_idx = discretize_state(obs)
            
            env_eval.render() 

            if terminated or truncated:
                break

        retornos.append(total_reward)

    env_eval.close()
    return np.mean(retornos), np.std(retornos)

# ===========================================================
# EJECUCION PRINCIPAL
# ===========================================================

Q = defaultdict(lambda: np.zeros(N_ACTIONS))
Returns_count = defaultdict(lambda: np.zeros(N_ACTIONS))
policy = create_initial_policy()

env = gym.make(ENV_NAME, render_mode=None)

returns = sarsa_control(env)

# Crear carpeta 'resultados'
save_dir = 'resultados'
os.makedirs(save_dir, exist_ok=True)

# Plot resultados
plt.plot(returns)
plt.xlabel('Episodios')
plt.ylabel('Retorno total por episodio')
plt.title('Monte Carlo Control en Pendulum-v1')

# Guardar el gráfico
plot_path = os.path.join(save_dir, 'grafico_retornos_sarsa.png')
plt.savefig(plot_path)
plt.close()

print(f"\nGráfico de retornos guardado en {plot_path}")


# Politica determinista final (argmax)
policy_det = {s: np.argmax(a) for s,a in Q.items()}

# Guardar politica
with open('policy_final_sarsa.pkl', 'wb') as f:
    pickle.dump(policy_det, f)

print("\nEntrenamiento finalizado. Politica guardada en policy_final_sarsa.pkl")

# ===========================================================
# PRUEBA DE LA POLÍTICA ENTRENADA CON RENDERIZADO
# ===========================================================
media, std = evaluar_politica(policy_det, n_episodios=100)
print(f"Retorno medio: {media:.2f}, desviación estándar: {std:.2f}")
