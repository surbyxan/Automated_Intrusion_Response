# from time import sleep
from CybORG.Agents import BaseAgent
from CybORG.Shared.Actions import Sleep, Restore, Remove
from .Offsets import Offsets

class ArgusV8(BaseAgent):


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
        Adaptive Blue Agent that adjusts response speed based on subnet-level threat.
        """
        # --- 1. Threat Assessment ---
        # Detect if Red has discovered any Enterprise IPs (Pivot risk)
        enterprise_under_threat = False
        for ent_host in ['Enterprise0', 'Enterprise1', 'Enterprise2']:
            ent_offset = Offsets.VECTOR_HOST_OFFSETS[ent_host]
            if observation[ent_offset + 0]: # Known bit is toggled
                enterprise_under_threat = True
                break

        # --- 2. High Priority Defense (Operational/Enterprise) ---
        for host_name in Offsets.PRIORITY_ORDER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]
            
            # Immediate response for high-value targets
            if observation[offset + 3] == 1: # Privileged
                return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            elif observation[offset + 2] == 1: # User
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        # --- 3. Adaptive User Host Defense ---
        for host_name in Offsets.PRIORITY_ORDER_USER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if observation[offset + 3] == 1: # Privileged
                # Adaptive Delay: 
                # If Enterprise is under threat, restore immediately (0 delay).
                # Otherwise, use a short 1-turn delay to attempt process removal/decoy deployment turns.
                max_delay = 0 if enterprise_under_threat else 1
                
                if self.UserRestoreCounter[host_name] < max_delay:
                    self.UserRestoreCounter[host_name] += 1
                else:
                    self.UserRestoreCounter[host_name] = 0
                    return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            
            elif observation[offset + 2] == 1: # User
                # Remove user-level sessions immediately
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        # --- 4. Decoy Deployment ---
        # Priority 1: Deploy decoys on hosts that are KNOWN but not yet SCANNED (The Strategic Window)
        for i, (host_name, decoy, decoy_flag) in enumerate(self.DecoyPriority):
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]
            if not observation[offset + 1] and observation[offset + 0]: 
                if not decoy_flag:
                    self.DecoyPriority[i] = (host_name, decoy, 1)
                    return Offsets.ACTION_OFFSETS[decoy] + Offsets.ACTION_HOST_OFFSET[host_name]

        # Priority 2: Deploy remaining decoys to fill slots/detect re-scans
        for i, (host_name, decoy, decoy_flag) in enumerate(self.DecoyPriority):
            if not decoy_flag:
                self.DecoyPriority[i] = (host_name, decoy, 1)
                return Offsets.ACTION_OFFSETS[decoy] + Offsets.ACTION_HOST_OFFSET[host_name]

        return 0

    def end_episode(self):
        # Reset counters for the next episode
        for host in self.UserRestoreCounter:
            self.UserRestoreCounter[host] = 0
        # Note: self.DecoyPriority should ideally be reset if the class isn't re-instantiated
        # but the current benchmark script re-instantiates the agent class per episode.
        pass
