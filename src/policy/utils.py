import random 
import numpy as np
import torch 


SEED =42

def set_seed(seed=SEED):
    """ Set random seed for reproducibility across numpy and Pythorch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

