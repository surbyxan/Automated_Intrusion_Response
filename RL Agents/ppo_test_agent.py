import sys
import os
# Add project root to sys.path so we can find RL_Agent
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from RL_Agent.cage2_ppo import DecoyState
from find_latest_model import find_latest_model

class PPOAgent:
    def __init__(self, model_path="ppo_test_cage2"):
        self.name = "PPOShapingTestAgent"
        
        # If model_path is a directory, find the latest model inside it
        if os.path.isdir(model_path):
            actual_path = find_latest_model(model_path)
            if actual_path is None:
                raise FileNotFoundError(f"No models found in directory: {model_path}")
        else:
            actual_path = model_path
            
        print(f"Loading PPO model from: {actual_path}")
        self.model = PPO.load(actual_path, device="cpu")
        self.decoy_state = DecoyState()

    def get_action(self, observation, _action_space):
        obs = self.decoy_state.get_state(observation)

        action, _states = self.model.predict(obs)
        self.decoy_state.do_action(action)
        # self.env.render("human")
        return action

    def end_episode(self):
        self.decoy_state.reset()

# display result db:
#  optuna-dashboard sqlite:///optuna_cage2_ppo.db --port 8080
