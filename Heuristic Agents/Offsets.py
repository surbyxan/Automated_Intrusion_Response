# from time import sleep
class Offsets():

    # ----------------------------- Observation space ---------------------------
    # Mapping of hostnames to their starting index in the 52-bit observation vector
    VECTOR_HOST_OFFSETS = {
        'Defender': 0, 'Enterprise0': 4, 'Enterprise1': 8, 'Enterprise2': 12,
        'Op_Host0': 16, 'Op_Host1': 20, 'Op_Host2': 24, 'Op_Server0': 28,
        'User0': 32, 'User1': 36, 'User2': 40, 'User3': 44, 'User4': 48
    }

    # Priority order for checking and defending hosts
    PRIORITY_ORDER = [
        # Operational Subnet
        'Op_Server0', 
        # Enterprise Subnet
        'Enterprise2', 'Enterprise0', 'Enterprise1',
        # Operational Hosts 
        'Op_Host0', 'Op_Host1', 'Op_Host2',
    ] 

    PRIORITY_ORDER_USER = [
        # User Subnet
        'User1', 'User2', 'User3', 'User4'
    ] 

    # ------------------------------ Action space --------------------------------
    # Offset for numbered action space
    ACTION_OFFSETS = {
        'sleep': 0, 'remove': 15, 'decoy_apache': 28, 'decoy_femitter': 41, 'decoy_haraka': 54, 'decoy_smss': 67,
        'decoy_sshd': 80, 'decoy_svchost': 93, 'decoy_tomcat': 106, 'decoy_vsftpd': 119, 'restore': 132
    }

    # Sub-offsets for hosts in action space 
    ACTION_HOST_OFFSET = {
        'Enterprise0': 1, 'Enterprise1': 2, 'Enterprise2': 3, 'Op_Host0': 4,
        'Op_Host1': 5, 'Op_Host2': 6, 'Op_Server0': 7,
        'User1': 9, 'User2': 10, 'User3': 11, 'User4': 12

    }
    # User 0 not in list since it is starting point
    # Defender not in list since it is 'invincible'
    #'Op_Host0', 'Op_Host1', 'Op_Host2',
