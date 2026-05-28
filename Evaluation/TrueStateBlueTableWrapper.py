from copy import deepcopy
from prettytable import PrettyTable
import numpy as np

from CybORG.Shared.Results import Results
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import TrueTableWrapper


class TrueStateBlueTableWrapper(BaseWrapper):
    def __init__(self, env=None, agent=None, output_mode='table'):
        super().__init__(env, agent)
        self.env = TrueTableWrapper(env=env, agent=agent)
        self.agent = agent

        self.baseline = None
        self.output_mode = output_mode

    def reset(self, agent='Blue'):
        result = self.env.reset(agent)
        obs = result.observation
        if agent == 'Blue':
            obs = self.observation_change(obs)
        result.observation = obs
        return result

    def step(self, agent=None, action=None) -> Results:
        result = self.env.step(agent, action)
        obs = result.observation
        if agent == 'Blue':
            obs = self.observation_change(obs)
        result.observation = obs
        result.action_space = self.action_space_change(result.action_space)
        return result

    def get_table(self):
        return self.env.get_table()

    def observation_change(self, observation):
        obs = deepcopy(observation)
        success = obs['success']

        del obs['success']

        if self.output_mode == 'table':
            return self.get_table()
        elif self.output_mode == 'raw':
            return observation
        elif self.output_mode == 'vector':
            return self._create_vector(success)
        else:
            raise NotImplementedError('Invalid output_mode for BlueTableWrapper')

    def _create_vector(self, success):
        table = self.env._create_true_table()._rows

        proto_vector = []
        for row in table:
            # Known
            known = row[3]
            value = [1] if known else [0]
            proto_vector.extend(value)

            # Scanned
            scanned = row[4]
            value = [1] if scanned else [0]
            proto_vector.extend(value)

            # Access
            access = row[5]
            if access == 'None':
                value = [0, 0]
            elif access == 'User':
                value = [1, 0]
            elif access == 'Privileged':
                value = [1, 1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

        return np.array(proto_vector)

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        if agent == 'Blue' and self.output_mode == 'table':
            output = self.get_table()
        else:
            output = self.get_attr('get_observation')(agent)

        return output

    def get_agent_state(self, agent: str):
        return self.get_attr('get_agent_state')(agent)

    def get_action_space(self, agent):
        return self.env.get_action_space(agent)

    def get_last_action(self, agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()
