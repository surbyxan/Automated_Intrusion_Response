# from time import sleep
from CybORG.Agents import BaseAgent
from CybORG.Shared.Actions import Sleep, Restore, Remove
from .Offsets import Offsets

class ArgusXI(BaseAgent):


    def __init__(self):
        super().__init__()
        self.name = "BlueArgusAgentXI"
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
        # same as V4
        self.DecoyPriority = [

            # High value decoys that most likely takes the 75% probability slot
            ('Enterprise0', 'decoy_femitter', 0),
            ('Enterprise1', 'decoy_femitter', 0),
            ('Enterprise2', 'decoy_femitter', 0),
            ('User4', 'decoy_femitter', 0),
            ('User2', 'decoy_femitter', 0),
            ('User3', 'decoy_femitter', 0),
            ('Op_Server0', 'decoy_femitter', 0), 
            ('Enterprise0', 'decoy_haraka', 0), 
            ('Enterprise1', 'decoy_haraka', 0), 
            ('Enterprise2', 'decoy_haraka', 0), 
            ('Op_Server0', 'decoy_haraka', 0), 
            ('User1', 'decoy_apache', 0),

            # Extra actions to fill out the 25% cut of non-top rated 
            # actions to minimize red exploatation success.
            ('Enterprise0', 'decoy_apache', 0), # new
            ('Enterprise0', 'decoy_tomcat', 0), # new
            ('Op_Server0', 'decoy_tomcat', 0), 
            ('Op_Server0', 'decoy_apache', 0), 
            ('User2', 'decoy_apache', 0),
            ('User3', 'decoy_sshd', 0), # new
            ('User1', 'decoy_tomcat', 0),
            ('User2', 'decoy_tomcat', 0),
            ('User1', 'decoy_smss', 0),
            ('User2', 'decoy_sshd', 0),
            ('User1', 'decoy_svchost', 0),

            ('Op_Host0', 'decoy_femitter', 0), 
            ('Op_Host1', 'decoy_femitter', 0), 
            ('Op_Host2', 'decoy_femitter', 0), 
            ('Op_Host0', 'decoy_haraka', 0), 
            ('Op_Host1', 'decoy_haraka', 0), 
            ('Op_Host2', 'decoy_haraka', 0), 
            ('Op_Host0', 'decoy_apache', 0), 
            ('Op_Host1', 'decoy_apache', 0),
            ('Op_Host2', 'decoy_apache', 0),
            ('Op_Host0', 'decoy_tomcat', 0),
            ('Op_Host1', 'decoy_tomcat', 0),    
            ('Op_Host2', 'decoy_tomcat', 0),


        ]
        # Counter to track turns since last restore action for 
        # user hosts to prioritize redeploying decoys on next turn.
        self.HostTurnCounter = {
            'User1':0, 'User2':0, 'User3':0, 'User4':0,
            'Enterprise0':0, 'Enterprise1':0, 'Enterprise2':0,
            'Op_Server0':0,
            'Op_Host0':0, 'Op_Host1':0, 'Op_Host2':0
        }

    def get_action(self, observation, action_space):
        """
        Scans hosts in priority order and restores the first compromised host found.
        """
       # print(self.DecoyPriority)

        # increment turn counter for all hosts
        for host_name in self.HostTurnCounter:
            self.HostTurnCounter[host_name] += 1

        # Restore or remove for enterprise or operational subnet
        for host_name in Offsets.PRIORITY_ORDER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]
            
            # Access bits are at index offset+2 and offset+3
            # [1, 0] = User, [1, 1] = Privileged
            # If the first access bit (offset+2) is 1, the host is compromised.
            if observation[offset + 3] == 1:
                for i, (decoy_host, decoy, decoy_flag) in enumerate(self.DecoyPriority):
                    if decoy_host == host_name:
                        self.DecoyPriority[i] = (decoy_host, decoy, 0)
                self.HostTurnCounter[host_name] = 0
                return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            elif observation[offset + 2] == 1:
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        # Restore or remove for user hosts
        for host_name  in Offsets.PRIORITY_ORDER_USER:
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if observation[offset + 3] == 1:
                # Wait 9 turns before restoring 
                # to not waste restore action each time a host in user is taken over. 
                if self.UserRestoreCounter[host_name] < 4:
                    self.UserRestoreCounter[host_name] += 1
                else:
                    self.UserRestoreCounter[host_name] = 0
                    for i, (decoy_host, decoy, decoy_flag) in enumerate(self.DecoyPriority):
                        if decoy_host == host_name:
                            self.DecoyPriority[i] = (decoy_host, decoy, 0)
                    self.HostTurnCounter[host_name] = 0
                    return Offsets.ACTION_OFFSETS['restore'] + Offsets.ACTION_HOST_OFFSET[host_name]
            elif observation[offset + 2] == 1:
                return Offsets.ACTION_OFFSETS['remove'] + Offsets.ACTION_HOST_OFFSET[host_name]

        
        # Priority decoys that is expected to be included in next 'portscan' 
        # (netscanned but not portscanned)
        for i, (host_name, decoy, decoy_flag) in enumerate(self.DecoyPriority):
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if self.HostTurnCounter[host_name] <= 2:    
                if not observation[offset + 1] and observation[offset + 0]: # if not yet portscanned and known
                    if not decoy_flag:
                        self.DecoyPriority[i] = (host_name, decoy, 1)
                        return Offsets.ACTION_OFFSETS[decoy] + Offsets.ACTION_HOST_OFFSET[host_name]
                
        # TODO: loop for high priority next-turn decoy deployment after host restoration.
        for i, (host_name, decoy, decoy_flag) in enumerate(self.DecoyPriority):
            offset = Offsets.VECTOR_HOST_OFFSETS[host_name]

            if self.HostTurnCounter[host_name] <= 2: 
                if not decoy_flag:
                    self.DecoyPriority[i] = (host_name, decoy, 1)
                    return Offsets.ACTION_OFFSETS[decoy] + Offsets.ACTION_HOST_OFFSET[host_name]

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

            # if not observation[offset + 1]: # if not yet portscanned 
            if not decoy_flag:
                self.DecoyPriority[i] = (host_name, decoy, 1)
                return Offsets.ACTION_OFFSETS[decoy] + Offsets.ACTION_HOST_OFFSET[host_name]


        # If no hosts are compromised and all decoys are deployed, Sleep
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
