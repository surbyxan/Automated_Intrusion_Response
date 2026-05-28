import sys
import os
import subprocess
import inspect
import time
from statistics import mean, stdev

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gymnasium
    sys.modules['gym'] = gymnasium
except ImportError:
    print("gymnasium not found, make sure to install it with 'pip install gymnasium'")
    

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
# from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
# from heuristic_agents.StaticAgentV2 import StaticAgentV2
# from heuristic_agents.ArgusV0 import ArgusV0
# from heuristic_agents.ArgusV1 import ArgusV1
# from heuristic_agents.ArgusV2 import ArgusV2
# from heuristic_agents.ArgusV3 import ArgusV3
# from heuristic_agents.ArgusV4 import ArgusV4
# from heuristic_agents.ArgusV5 import ArgusV5
# from heuristic_agents.ArgusV6 import ArgusV6
# from heuristic_agents.ArgusV7 import ArgusV7
# from heuristic_agents.ArgusV8 import ArgusV8
# from heuristic_agents.ArgusV9 import ArgusV9
# from heuristic_agents.ArgusX import ArgusX
# from heuristic_agents.ArgusXI import ArgusXI

# from RL_Agent.cage2_dql import Cage2DQL
# from RL_Agent.ppo_test_agent import PPOAgent
# from RL_Agent.dql_test_agent import DQLAgent
# from RL_Agent.ppo_test_agent import PPOAgent
from RL_Agent.ppo_test_meander_only import PPOAgent
from RL_Agent.dql_test_agent import DQLAgent


from TrueStateChallengeWrapper import TrueStateChallengeWrapper

MAX_EPS = 100
agent_name = 'Blue'


def wrap(env):
    return TrueStateChallengeWrapper(env=env, agent_name='Blue')


if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'

    # Check for command line argument for model path
    model_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Change this line to load your agent
    # agent = BlueLoadAgent()
    # agent = StaticAgentV2()
    # agent = ArgusV0()
    # agent = ArgusV1()
    # agent = ArgusV2()
    # agent = ArgusV3()
    # agent = ArgusV4()
    # agent = ArgusV5()
    # agent = ArgusV6()
    # agent = ArgusV7()
    # agent = ArgusV8()
    # agent = ArgusV9()
    # agent = ArgusX()
    # agent = ArgusXI()
    # agent = Cage2DQL()
    # agent = PPOAgent()
    # agent = DQLAgent()
    
    if model_path:
        # Determine if we should use DQL or PPO based on path name
        if 'dqn' in model_path.lower() or 'dql' in model_path.lower():
            agent = DQLAgent(model_path=model_path)
        else:
            agent = PPOAgent(model_path=model_path)
            
        # Clean the model_path to use it in a filename (remove slashes)
        model_tag = os.path.basename(model_path.rstrip('/'))
    else:
        agent = PPOAgent()
        model_tag = "default"


    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = './Evaluation_result/' + time.strftime("%Y%m%d_%H%M%S") + f'_{model_tag}_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)
                    observation, rew, terminated, truncated, info = wrapped_cyborg.step(action)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward):.2f} with a standard deviation of {stdev(total_reward):.2f}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
