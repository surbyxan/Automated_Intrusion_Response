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
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
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
        if action < 28 or (action >= 119 and action < 132): return
        if 132 <= action <= 144:
            host_id = action - 132
            if host_id in [0, 8]: return
            host_id = self.ACTION_ID_TO_VECTOR_IDX[host_id]
            offset = host_id * self.num_decoys
            self.decoys[offset: offset + self.num_decoys] = [0] * self.num_decoys
        if 28 <= action <= 131:
            relative_action = action - 28
            host_id = (relative_action % 13)
            decoy_id = relative_action // 13
            if host_id in [0, 8]: return
            host_id = self.ACTION_ID_TO_VECTOR_IDX[host_id]
            if decoy_id in self.DECOY_ID_COMPATABILITY.get(host_id, []):
                self.decoys[host_id * self.num_decoys + decoy_id] = 1

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
        self.decoy_state.do_action(action)
        return self.decoy_state.get_state(obs), reward, terminated, truncated, info

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
    EXPERIMENT_NAME = "DQN_regular"          # E.g., "DQN_Base", "DQN_Penalty"
    # OLD: RESUME_TRAINING = False                  # True if it should pick up from crash
    RESUME_TRAINING = False
    TOTAL_TIMESTEPS = 5_000_000

    # dir stuff
    CHECKPOINT_DIR = f"./checkpoints/{EXPERIMENT_NAME}/"
    TENSORBOARD_DIR = "./tensorboard_logs/"
    LATEST_MODEL_PATH = find_latest_model(CHECKPOINT_DIR)

    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"=== Starting Experiment: {EXPERIMENT_NAME} ===")
    
    # env setup
    env_funcs = [
        make_env(B_lineAgent), 
        make_env(RedMeanderAgent), 
        make_env(SleepAgent)
    ]
    vec_env = SubprocVecEnv(env_funcs)

    # initialize
    if RESUME_TRAINING and LATEST_MODEL_PATH and os.path.exists(LATEST_MODEL_PATH):
        print(f"-> Resuming previous training from {LATEST_MODEL_PATH}...")
        model = DQN.load(LATEST_MODEL_PATH, env=vec_env, device="cpu", tensorboard_log=TENSORBOARD_DIR)
    else:
        if RESUME_TRAINING:
            print("-> Warning: Resume set to True, but no previous model found. Starting fresh.")
        else:
            print("-> Starting fresh training run...")
            
        # best architecture from optune
        policy_kwargs = dict(net_arch=[256, 256])
        
        # Inject the exact Optuna parameters
        model = DQN("MlpPolicy", vec_env, 
                    learning_rate=0.000916673,
                    buffer_size=200000,                 
                    learning_starts=10000,
                    batch_size=500,
                    gamma=0.99,                           
                    train_freq=16,
                    gradient_steps=1,
                    target_update_interval=1000,
                    exploration_fraction=0.15,
                    exploration_final_eps=0.05,

                    policy_kwargs=policy_kwargs,
                    verbose=1,               
                    device="cpu",
                    tensorboard_log=TENSORBOARD_DIR)

    # setup checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000, 1), 
        save_path=CHECKPOINT_DIR,
        name_prefix=f'{EXPERIMENT_NAME}_step'
    )

    print(f"-> Target Timesteps: {TOTAL_TIMESTEPS:,}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=checkpoint_callback, 
        reset_num_timesteps=not RESUME_TRAINING,
        tb_log_name=EXPERIMENT_NAME
    )

    # save the Final Model
    model.save(LATEST_MODEL_PATH)
    print(f"=== Experiment {EXPERIMENT_NAME} Completed and Saved! ===")
    
    vec_env.close()
