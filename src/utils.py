import torch
import numpy as np
import random

# sets seed of all libraries we're using 
# so we get consistent results across tests
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
