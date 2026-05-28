# from time import sleep
from CybORG.Agents import BaseAgent
from CybORG.Shared.Actions import Sleep, Restore, Remove
from .Offsets import Offsets

class ArgusV1(BaseAgent):


    def __init__(self):
        super().__init__()
        self.name = "BlueArgusAgent"
        self.UserRestoreCounter = {
            'User1':0, 'User2':0, 'User3':0, 'User4':0
        }
        print(f"Testing offset:{Offsets.PRIORITY_ORDER[0]}")

    def get_action(self, observation, action_space):
        """
        Scans hosts in priority order and restores the first compromised host found.
        """
        for host_name in Offsets.PRIORITY_ORDER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]
            
            # Access bits are at index offset+2 and offset+3
            # [1, 0] = User, [1, 1] = Privileged
            # If the first access bit (offset+2) is 1, the host is compromised.
            if observation[offset + 3] == 1:
                return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            elif observation[offset + 2] == 1:
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        for host_name  in Offsets.PRIORITY_ORDER_USER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if observation[offset + 3] == 1:
                # Wait 9 turns before restoring 
                # to not waste restore action each time a host in user is taken over. 
                if self.UserRestoreCounter[host_name] < 6:
                    self.UserRestoreCounter[host_name] += 1
                else:
                    self.UserRestoreCounter[host_name] = 0
                    return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            elif observation[offset + 2] == 1:
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        # TODO: add decoys

        # If no hosts are compromised, Sleep
        return 0

    def end_episode(self):
        pass


    # defender = observation[0:4]
    # ent0 = observation[4:8]
    # ent1 = observation[8:12]
    # ent2 = observation[12:16]
    # op_host0 = observation[16:20]
    # op_host1 = observation[20:24]
    # op_host2 = observation[24:28]
    # op_server0 = observation[28:32]
    # user0 = observation[32:36]
    # user1 = observation[36:40]
    # user2 = observation[40:44]
    # user3 = observation[44:48]
    # user4 = observation[48:52]
