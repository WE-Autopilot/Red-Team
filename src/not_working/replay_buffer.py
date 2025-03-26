##############################
##      REPLAY BUFFER       ##
##############################
import random
from collections import deque

import numpy as np

class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples for off-policy RL.
    """
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: bool):
        self.buffer.append((s, a, r, ns, d))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d
    
    def __len__(self) -> int:
        return len(self.buffer)