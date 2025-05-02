import gymnasium as gym
import numpy as np
from collections import defaultdict
from itertools import product
import pandas as pd

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

ENV_NAME = 'Pendulum-v1'
MAX_STEPS = 200
N_EPISODIOS = 10000
EVAL_EPISODIOS = 10
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01

# Grilla de hiperparámetros
ANGLE_BINS_LIST = [15, 30, 40]
VELOCITY_BINS_LIST = [15, 30, 40]
ACTIONS_LIST = [11, 17]
ALPHAS = [0.05, 0.1]
DECAYS = [0.995, 0.999]

# Espacio de acciones continuas
ACTION_MIN, ACTION_MAX = -2.0, 2.0
ANGLE_MIN, ANGLE_MAX = -np.pi, np.pi
VEL_MIN, VEL_MAX = -8.0, 8.0


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def discretize_state(obs, angle_bins, vel_bins):
    cos_theta, sin_theta, theta_dot = obs
    theta = np.arctan2(sin_theta, cos_theta)
    angle_idx = np.digitize(theta, angle_bins) - 1
    vel_idx = np.digitize(theta_dot, vel_bins) - 1
    angle_idx = np.clip(angle_idx, 0, len(angle_bins)-2)
    vel_idx = np.clip(vel_idx, 0, len(vel_bins)-2)
    return (angle_idx, vel_idx)

def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

def q_learning(env, angle_bins, vel_bins, action_list, alpha, epsilon_decay):
    Q = defaultdict(lambda: np.zeros(len(action_list)))
    epsilon = EPSILON_START
    returns = []

    for ep in range(N_EPISODIOS):
        obs, _ = env.reset()
        state = discretize_state(obs, angle_bins, vel_bins)
        total_reward = 0

        for _ in range(MAX_STEPS):
            a_idx = epsilon_greedy(Q, state, epsilon, len(action_list))
            action = [action_list[a_idx]]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_obs, angle_bins, vel_bins)

            # Q-Learning update
            best_next = np.max(Q[next_state])
            td_target = reward + GAMMA * best_next
            Q[state][a_idx] += alpha * (td_target - Q[state][a_idx])

            state = next_state
            total_reward += reward

            if done:
                break

        returns.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)

    # Política final determinista
    policy = {s: np.argmax(a) for s, a in Q.items()}
    return Q, policy, returns


def evaluar_politica(env, policy, action_list, angle_bins, vel_bins):
    rewards = []

    for _ in range(EVAL_EPISODIOS):
        obs, _ = env.reset()
        state = discretize_state(obs, angle_bins, vel_bins)
        total_reward = 0

        for _ in range(MAX_STEPS):
            a_idx = policy.get(state, np.random.choice(len(action_list)))
            action = [action_list[a_idx]]
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize_state(obs, angle_bins, vel_bins)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    return np.mean(rewards)


# ============================================
# GRID SEARCH PRINCIPAL
# ============================================

env = gym.make(ENV_NAME, render_mode=None)
resultados = []

for n_angle, n_vel, n_act, alpha, decay in product(ANGLE_BINS_LIST, VELOCITY_BINS_LIST, ACTIONS_LIST, ALPHAS, DECAYS):
    angle_bins = np.linspace(ANGLE_MIN, ANGLE_MAX, n_angle+1)
    vel_bins = np.linspace(VEL_MIN, VEL_MAX, n_vel+1)
    action_list = np.linspace(ACTION_MIN, ACTION_MAX, n_act)

    Q, policy, train_returns = q_learning(env, angle_bins, vel_bins, action_list, alpha, decay)
    eval_return = evaluar_politica(env, policy, action_list, angle_bins, vel_bins)

    resultados.append({
        "angle_bins": n_angle,
        "vel_bins": n_vel,
        "n_actions": n_act,
        "alpha": alpha,
        "decay": decay,
        "eval_return": eval_return
    })

    print(f"Combinación ({n_angle}, {n_vel}, {n_act}, {alpha}, {decay}) → Retorno: {eval_return:.2f}")

env.close()

# ============================================
# RESULTADOS ORDENADOS
# ============================================

df = pd.DataFrame(resultados)
df_sorted = df.sort_values(by="eval_return", ascending=False)

print("\n=== TOP COMBINACIONES ===")
print(df_sorted.head(10).to_string(index=False))

mejor = df_sorted.iloc[0]
print(f"\n🏆 Mejor combinación:\n{mejor.to_dict()}")


"""=== TOP COMBINACIONES ===
 angle_bins  vel_bins  n_actions  alpha  decay  eval_return
         30        15         21   0.05  0.997  -669.267171
         24        15         17   0.20  0.999  -778.682203
         20        15         17   0.10  0.999  -801.511425
         20        15         17   0.20  0.999  -822.714039
         24        20         17   0.20  0.999  -827.980718
         20        15         25   0.20  0.995  -869.826213
         20        30         17   0.10  0.999  -873.069911
         20        30         21   0.20  0.997  -880.892644
         24        15         17   0.20  0.995  -904.481918
         30        20         25   0.20  0.997  -905.876540

🏆 Mejor combinación:
{'angle_bins': 30.0, 'vel_bins': 15.0, 'n_actions': 21.0, 'alpha': 0.05, 'decay': 0.997, 'eval_return': -669.2671709118757}"""