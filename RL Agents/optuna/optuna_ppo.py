import sys
import os
import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
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
        
        # Original CybORG space (52) + Decoy space (77) = 129
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
    """Creates a wrapped environment for a specific Red Agent, terminating at max_steps."""
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
def optimize_ppo(trial):
    """Objective function for Optuna to evaluate hyperparameters."""
    
    # 1. Suggest Standard Hyperparameters
    n_steps = trial.suggest_categorical("n_steps", [5000, 10000, 15000])
    # Batch sizes adjusted to be factors of (5000*3), (10000*3), and (15000*3)
    batch_size = trial.suggest_categorical("batch_size", [60, 120, 250, 500])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.01, log=True)
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15])
    
    # 2. Suggest PPO-Specific Parameters
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    # 3. Suggest Network Architecture Capacity
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    elif net_arch_choice == "medium":
        net_arch = dict(pi=[256, 256], vf=[256, 256])
    else: # large
        net_arch = dict(pi=[512, 256, 128], vf=[512, 256, 128])
        
    policy_kwargs = dict(net_arch=net_arch)

    # Ensure batch_size is a factor of (n_steps * n_envs). n_envs is 3 here.
    if (n_steps * 3) % batch_size != 0:
        print(f"Trial {trial.number} pruned due to math mismatch: ({n_steps}*3) % {batch_size} != 0")
        raise optuna.exceptions.TrialPruned() 

    # 4. Create the Vectorized Environment for the trial
    env_funcs = [
        make_env(B_lineAgent, max_steps=100),
        make_env(RedMeanderAgent, max_steps=100),
        make_env(SleepAgent, max_steps=100)
    ]
    vec_env = SubprocVecEnv(env_funcs) 

    # Create Model with TensorBoard logging
    model = PPO("MlpPolicy", vec_env, 
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=0.99,                 # Hardcoded to 0.99
                gae_lambda=gae_lambda,     
                clip_range=clip_range,     
                policy_kwargs=policy_kwargs, 
                verbose=0,
                device="cpu",
                tensorboard_log="./tensorboard_logs/ppo_cage2_optuna") 

    # Train for 500k steps
    try:
        model.learn(total_timesteps=500_000, tb_log_name=f"trial_{trial.number}")
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        vec_env.close()
        raise optuna.exceptions.TrialPruned() 

    # Evaluate the model 
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=15)
    
    # Store extra metadata in the study
    trial.set_user_attr("std_reward", std_reward)
    
    vec_env.close()
    return mean_reward

# ------------------------- 5. Main Execution -------------------------
if __name__ == '__main__':
    print("Starting Optuna Hyperparameter Search for PPO...")
    
    # Save the study to an SQLite database.
    study = optuna.create_study(
        direction="maximize", 
        study_name="ppo_cage2_optimization", 
        storage="sqlite:///optuna_cage2_ppo.db", 
        load_if_exists=True
    )
    
    # Run trials
    study.optimize(optimize_ppo, n_trials=75, n_jobs=2) 
    
    # Export results to CSV for the final table
    df = study.trials_dataframe()
    df.to_csv("optuna_ppo_results_table.csv", index=False)
    
    print("\n==================================")
    print(f"Optimization Complete! Results saved to optuna_ppo_results_table.csv")
    print("Best Hyperparameters:", study.best_params)
    print("Best Reward:", study.best_value)
    print("==================================\n")


# ==================================
# Optimization Complete!
# Best Hyperparameters: {'learning_rate': 0.00015960458201276067, 'ent_coef': 0.006611636036699288, 'n_steps': 4096, 'batch_size': 64, 'n_epochs': 5, 'gae_lambda': 0.9223883969368327, 'clip_range': 0.3, 'net_arch': 'medium'}
# Best Reward: -17.06000000000001
# ==================================
