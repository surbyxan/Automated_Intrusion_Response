import sys
import os
import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Add project root to sys.path to allow absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.modules['gym'] = gym
sys.path.append(os.path.join(project_root, 'CustomScenario'))

# CybORG Imports
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CustomScenario.TrueStateChallengeWrapper import TrueStateChallengeWrapper
import inspect

# ------------------------- 1. Decoy State Logic -------------------------
class DecoyState():
    def __init__(self):
        # 11 targetable hosts, 7 decoy types each = 77 bits
        self.decoys = [0] * 77
        self.num_decoys = 7
        self.ACTION_ID_TO_VECTOR_IDX = {
            0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 8, 9: 7, 10: 8, 11: 9, 12: 10
        }
        self.DECOY_ID_COMPATABILITY = {
            7: [6, 0, 5, 3], 8: [1, 6, 0, 4], 9: [1,4], 10: [1], 
            0: [1,2,6,0], 1: [1,2], 2: [1,2], 3: [1,2,6,0], 
            4: [1,2,6,0], 5: [1,2,6,0], 6: [1,2,6,0]
        }

    def reset(self):
        self.decoys = [0] * 77

    def get_state(self, observation):
        return np.concatenate([observation, self.decoys])

    def do_action(self, action):	
        if action < 28 or (action >= 119 and action < 132): 
            return
        if 132 <= action <= 144:
            host_id = action - 132
            if host_id in [0, 8]: 
                return
            host_id = self.ACTION_ID_TO_VECTOR_IDX[host_id]
            offset = host_id * self.num_decoys
            self.decoys[offset: offset + self.num_decoys] = [0] * self.num_decoys

        if 28 <= action <= 131:
            relative_action = action - 28
            host_id = (relative_action % 13)
            decoy_id = relative_action // 13
            if host_id in [0, 8]: 
                return
            host_id = self.ACTION_ID_TO_VECTOR_IDX[host_id]
            if decoy_id in self.DECOY_ID_COMPATABILITY.get(host_id, []):
                self.decoys[host_id * self.num_decoys + decoy_id] = 1

# ------------------------- 2. SB3 Environment Wrapper -------------------------
class CybORGDecoyWrapper(gym.Wrapper):
    """Integrates DecoyState with SB3 by defining the unified observation space."""
    def __init__(self, env):
        super().__init__(env)
        self.decoy_state = DecoyState()
        original_shape = self.env.observation_space.shape[0]
        new_shape = original_shape + 77
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(new_shape,), dtype=np.float32
        )

    def reset(self, **kwargs):
        self.decoy_state.reset()
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0] 
        return self.decoy_state.get_state(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.decoy_state.do_action(action)
        return self.decoy_state.get_state(obs), reward, terminated, truncated, info

# ------------------------- 3. Environment Factory -------------------------
def make_env(red_agent_class, max_steps=100):
    """Creates a wrapped environment for a specific Red Agent."""
    def _init():
        path = str(inspect.getfile(CybORG))[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        base_env = TrueStateChallengeWrapper(
            env=CybORG(path, 'sim', agents={'Red': red_agent_class}), 
            agent_name='Blue'
        )
        decoy_env = CybORGDecoyWrapper(base_env)
        time_env = TimeLimit(decoy_env, max_episode_steps=max_steps)
        return Monitor(time_env)
    return _init

# ------------------------- 4. Optuna Optimization -------------------------
def optimize_dqn(trial):
    """Objective function for Optuna to evaluate DQN hyperparameters."""
    
    # 1. Suggest Learning Rates & Buffer
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [60, 120, 250, 500])
    # Cap buffer size to 200,000 to prevent RAM crash during parallel 4-core execution
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000]) 
    
    # 2. Suggest DQN-Specific Exploration and Updating Dynamics
    # train_freq = trial.suggest_categorical("train_freq", [4, 8, 16])
    train_freq = trial.suggest_categorical("train_freq", [16, 32, 64])
    target_update_interval = trial.suggest_categorical("target_update_interval", [1000, 5000, 10000])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.7)
    exploration_final_eps = trial.suggest_categorical("exploration_final_eps", [0.01, 0.05])

    # 3. Suggest Network Architecture Capacity (DQN uses lists, not dicts)
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [128, 128]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else: # large
        net_arch = [512, 256, 128]
        
    policy_kwargs = dict(net_arch=net_arch)

    # 4. Create the Vectorized Environment
    env_funcs = [
        make_env(B_lineAgent, max_steps=100),
        make_env(RedMeanderAgent, max_steps=100),
        make_env(SleepAgent, max_steps=100)
    ]
    # Using DummyVecEnv is safer for DQN memory management during hyperparameter tuning
    # vec_env = DummyVecEnv(env_funcs) 
    vec_env = SubprocVecEnv(env_funcs) 

    # 5. Initialize DQN Model
    model = DQN("MlpPolicy", vec_env, 
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=10000,       # Seed buffer with 10k random actions
                batch_size=batch_size,
                gamma=0.99,                   # Hardcoded to 1.0 due to strict 100-step horizon
                train_freq=train_freq,
                gradient_steps=1,
                target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps,
                policy_kwargs=policy_kwargs, 
                verbose=0,
                device="cpu",
                tensorboard_log="./tensorboard_logs/dqn_cage2_optuna") 

    # Train for 500k steps
    try:
        model.learn(total_timesteps=500_000, tb_log_name=f"trial_{trial.number}")
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        vec_env.close()
        raise optuna.exceptions.TrialPruned() 

    # Evaluate the model (15 episodes = 5 vs B_line, 5 vs Meander, 5 vs Sleep)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=15)
    
    # Store extra metadata in the study
    trial.set_user_attr("std_reward", std_reward)
    
    vec_env.close()
    return mean_reward

# ------------------------- 5. Main Execution -------------------------
if __name__ == '__main__':
    print("Starting Optuna Hyperparameter Search for DQN...")
    
    # Save the study to a separate SQLite database.
    study = optuna.create_study(
        direction="maximize", 
        study_name="dqn_cage2_optimization", 
        storage="sqlite:///optuna_cage2_dqn.db", 
        load_if_exists=True
    )
    
    # Run 75 trials
    study.optimize(optimize_dqn, n_trials=75, n_jobs=2) 
    
    # Export results to CSV for the final table
    df = study.trials_dataframe()
    df.to_csv("optuna_dqn_results_table.csv", index=False)
    
    print("\n==================================")
    print(f"Optimization Complete! Results saved to optuna_dqn_results_table.csv")
    print("Best Hyperparameters:", study.best_params)
    print("Best Reward:", study.best_value)
    print("==================================\n")

# ==================================
# Optimization Complete!
# Best Hyperparameters: {'learning_rate': 0.0009562729409442042, 'batch_size': 128, 'buffer_size': 50000, 'train_freq': 16, 'target_update_interval': 10000, 'exploration_fraction': 0.4911220683707432, 'exploration_final_eps': 0.01, 'net_arch': 'large'}
# Best Reward: -20.679999999999968
# ==================================
