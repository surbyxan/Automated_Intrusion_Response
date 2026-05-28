import os
import warnings

# TensorFlow/CUDA C++ Warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Gym Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", module="gymnasium")
warnings.filterwarnings("ignore", module="gym")

import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Add project root to sys.path to allow absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.modules['gym'] = gym
sys.path.append(os.path.join(project_root, 'CustomScenario'))

# CybORG Imports
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CustomScenario.TrueStateChallengeWrapper import TrueStateChallengeWrapper
from find_latest_model import find_latest_model
import inspect

# ------------------------- Decoy State Logic -------------------------
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
        valid_action_penalty = 0
        if action < 28 or (action >= 119 and action < 132): 
            return valid_action_penalty
        if 132 <= action <= 144:
            host_id = action - 132
            if host_id in [0, 8]: 
                valid_action_penalty = -0.1
                return valid_action_penalty
            host_id = self.ACTION_ID_TO_VECTOR_IDX[host_id]
            offset = host_id * self.num_decoys
            self.decoys[offset: offset + self.num_decoys] = [0] * self.num_decoys

        if 28 <= action <= 131:
            relative_action = action - 28
            host_id = (relative_action % 13)
            decoy_id = relative_action // 13
            if host_id in [0, 8]: 
                valid_action_penalty = -0.1
                return valid_action_penalty
            host_id = self.ACTION_ID_TO_VECTOR_IDX[host_id]
            if decoy_id in self.DECOY_ID_COMPATABILITY.get(host_id, []):
                self.decoys[host_id * self.num_decoys + decoy_id] = 1
            else:
                valid_action_penalty = -0.1
                return valid_action_penalty
        return valid_action_penalty


# ------------------------- SB3 Environment Wrapper -------------------------
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
        custom_penalty = self.decoy_state.do_action(action)
        shaped_reward = reward + custom_penalty
        return self.decoy_state.get_state(obs), shaped_reward, terminated, truncated, info

# ------------------------- Custom Callbacks -------------------------
class FrequentLoggingCallback(BaseCallback):
    """
    Forces logging of episodic rewards at a higher frequency than the default
    PPO rollout window.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                rewards = [info['r'] for info in self.model.ep_info_buffer]
                lengths = [info['l'] for info in self.model.ep_info_buffer]
                self.logger.record("rollout/ep_rew_mean_high_res", np.mean(rewards))
                self.logger.record("rollout/ep_len_mean_high_res", np.mean(lengths))
                self.logger.dump(step=self.num_timesteps)
        return True

# ------------------------- Environment Factory -------------------------
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

# ------------------------- Main Execution -------------------------
if __name__ == '__main__':
    # experiment conf
    EXPERIMENT_NAME = "PPO_more_meander"          # E.g., "PPO_Base", "PPO_Penalty", "PPO_More_Meander"
    RESUME_TRAINING = False               # True if it should pick up from crash
    TOTAL_TIMESTEPS = 5_000_000
    
    # dir stuff
    CHECKPOINT_DIR = f"./checkpoints/{EXPERIMENT_NAME}/"
    TENSORBOARD_DIR = "./tensorboard_logs/"
    LATEST_MODEL_PATH = find_latest_model(CHECKPOINT_DIR)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"=== Starting Experiment: {EXPERIMENT_NAME} ===")
    
    # env setup
    env_funcs = [
        make_env(B_lineAgent), # make_env(B_lineAgent), make_env(B_lineAgent),
        make_env(RedMeanderAgent), make_env(RedMeanderAgent) #, make_env(RedMeanderAgent)
        # make_env(SleepAgent), # make_env(SleepAgent), make_env(SleepAgent)
    ]
    vec_env = SubprocVecEnv(env_funcs)

    # 4. initialize
    if RESUME_TRAINING and LATEST_MODEL_PATH and os.path.exists(LATEST_MODEL_PATH):
        print(f"-> Resuming previous training from {LATEST_MODEL_PATH}...")
        model = PPO.load(LATEST_MODEL_PATH, env=vec_env, device="cpu", tensorboard_log=TENSORBOARD_DIR)
    else:
        if RESUME_TRAINING:
            print("-> Warning: Resume set to True, but no previous model found. Starting fresh.")
        else:
            print("-> Starting fresh training run...")
            
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO("MlpPolicy", vec_env, 
                    learning_rate=0.00021288,
                    ent_coef=0.00040938,
                    n_steps=10000,
                    batch_size=60,
                    n_epochs=5,
                    gamma=0.99,               
                    gae_lambda= 0.9109549,
                    clip_range=0.300000,
                    policy_kwargs=policy_kwargs,
                    verbose=1,               
                    device="cpu",
                    tensorboard_log=TENSORBOARD_DIR) 
 

    # callbacks & training
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000, 1), 
        save_path=CHECKPOINT_DIR,       # Saves inside the isolated experiment folder
        name_prefix=f'{EXPERIMENT_NAME}_step'
    )
    
    logging_callback = FrequentLoggingCallback(log_freq=1000)

    print(f"-> Target Timesteps: {TOTAL_TIMESTEPS:,}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=[checkpoint_callback, logging_callback], 
        reset_num_timesteps=not RESUME_TRAINING, 
        tb_log_name=EXPERIMENT_NAME
    )

    # Save the Final "Latest" Model
    model.save(LATEST_MODEL_PATH)
    print(f"=== Experiment {EXPERIMENT_NAME} Completed and Saved! ===")
    
    vec_env.close()
