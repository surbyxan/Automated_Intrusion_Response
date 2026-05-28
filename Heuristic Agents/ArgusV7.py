# from time import sleep
from CybORG.Agents import BaseAgent
from CybORG.Shared.Actions import Sleep, Restore, Remove
from .Offsets import Offsets

class ArgusV7(BaseAgent):


    def __init__(self):
        super().__init__()
        self.name = "BlueArgusAgent"
        self.UserRestoreCounter = {
            'User1':0, 'User2':0, 'User3':0, 'User4':0
        }
        self.DecoysDeployed = { # [ femitter, haraka ]
            'Op_Server0': [0, 0], 'Enterprise2': [0, 0], 
            'Enterprise0':[0, 0], 'Enterprise1':[0, 0],
            'User1': [0], # [ haraka ]
            'User2': [0, 0, 0], # [ femitter, haraka, apache ] 
            'User3': [0], # [ femitter ] 
            'User4': [0], # [ femitter ] 
        }
        
        self.DecoyPriority = [
            # --- High Weight Decoys (Weight 7.0 / 6.0) ---
            # These occupy the 75% probability slot for Red's exploit selection.
            # Windows Hosts (Free Port 21)
            ('Enterprise1', 'decoy_femitter', 0),
            ('Enterprise2', 'decoy_femitter', 0),
            ('User2', 'decoy_femitter', 0),
            
            # Linux Hosts (Free Port 21)
            ('User3', 'decoy_vsftpd', 0),
            ('User4', 'decoy_vsftpd', 0),

            # Linux Hosts (Free Port 25)
            ('Enterprise0', 'decoy_haraka', 0),
            ('Op_Server0', 'decoy_haraka', 0),

            # --- Low Weight / Filler Decoys ---
            # These fill the 25% slot to minimize the chance of Red picking a real service.
            
            # Windows User1 (Free Ports 80, 443, 139, 3389 - Port 21 is occupied by real Femitter)
            ('User1', 'decoy_smss', 0),
            ('User1', 'decoy_svchost', 0),
            ('User1', 'decoy_tomcat', 0),
            ('User1', 'decoy_apache', 0),

            # Linux Hosts (Free Ports 80, 443)
            ('Enterprise0', 'decoy_tomcat', 0),
            ('Enterprise0', 'decoy_apache', 0),
            ('Op_Server0', 'decoy_tomcat', 0),
            ('Op_Server0', 'decoy_apache', 0),
            
            # Linux User3 (Port 22 is free)
            ('User3', 'decoy_sshd', 0),
        ]

    def get_action(self, observation, action_space):
        """
        Scans hosts in priority order and restores the first compromised host found.
        """
        # Restore or remove for enterprise or operational subnet
        for host_name in Offsets.PRIORITY_ORDER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]
            
            # Access bits are at index offset+2 and offset+3
            # [1, 0] = User, [1, 1] = Privileged
            # If the first access bit (offset+2) is 1, the host is compromised.
            if observation[offset + 3] == 1:
                return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            elif observation[offset + 2] == 1:
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        # Restore or remove for user hosts
        for host_name  in Offsets.PRIORITY_ORDER_USER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if observation[offset + 3] == 1:
                # Wait before restoring user hosts to save actions
                if self.UserRestoreCounter[host_name] < 4:
                    self.UserRestoreCounter[host_name] += 1
                else:
                    self.UserRestoreCounter[host_name] = 0
                    return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            elif observation[offset + 2] == 1:
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        # Priority decoys that is expected to be included in next 'portscan' 
        # (netscanned but not portscanned)
        for i, (host_name, decoy, decoy_flag) in enumerate(self.DecoyPriority):
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if not observation[offset + 1] and observation[offset + 0]: # if not yet portscanned and known
                if not decoy_flag:
                    self.DecoyPriority[i] = (host_name, decoy, 1)
                    return Offsets.ACTION_OFFSETS[decoy] + Offsets.ACTION_HOST_OFFSET[host_name]


        # Deploy remaining decoys even if portscanned or not netscanned
        for i, (host_name, decoy, decoy_flag) in enumerate(self.DecoyPriority):
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if not decoy_flag:
                self.DecoyPriority[i] = (host_name, decoy, 1)
                return Offsets.ACTION_OFFSETS[decoy] + Offsets.ACTION_HOST_OFFSET[host_name]


        # If no hosts are compromised, Sleep
        return 0

    def end_episode(self):
        pass
