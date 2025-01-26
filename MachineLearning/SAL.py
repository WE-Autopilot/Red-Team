import torch
import torch.nn as nn
import torch.nn.functional as F

class SAL(nn.module): 
    super(SAL, self).__init__()