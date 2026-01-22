import numpy as np

""" NSCL """
# IMG_MEAN = np.array([0.0531, 0.0531, 0.0531])
# IMG_STD = np.array([0.2037, 0.2145, 0.2066])
# 
# tcp normalization and gripper width normalization
# ACTION_TRANS_MAX = np.array([0.413985, 0.037210, 0.347438]) 
# ACTION_TRANS_MIN = np.array([0.056788, -0.273891, 0.023419]) 
# OBS_TRANS_MAX = np.array([0.413297, 0.037082, 0.347349]) 
# OBS_TRANS_MIN = np.array([0.061361, -0.274095, 0.025086]) 
# 
# OBS_GRIP_MAX = np.array([94.2])
# OBS_GRIP_MIN = np.array([-0.2])

""" MoAi """
IMG_MEAN=np.array([0.0286, 0.0286, 0.0286])
IMG_STD=np.array([0.1690, 0.1690, 0.1690])

# tcp normalization and gripper width normalization
ACTION_TRANS_MAX = np.array([0.413985, 0.081356, 0.349960]) 
ACTION_TRANS_MIN = np.array([-0.009628, -0.298933, 0.023419]) 
OBS_TRANS_MAX = np.array([0.413297, 0.082003, 0.347349]) 
OBS_TRANS_MIN = np.array([-0.009572, -0.300488, 0.025086]) 
OBS_GRIP_MAX = np.array([94.2])
OBS_GRIP_MIN = np.array([-0.2])


TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']

