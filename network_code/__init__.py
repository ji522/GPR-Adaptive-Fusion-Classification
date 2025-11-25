"""
GPR三分支物理引导网络
Three-Branch Physics-Guided Network for GPR Image Classification
"""

__version__ = "1.0.0"
__author__ = "GPR Research Team"

from .models import TriBranchNetwork
from .losses import SelectiveConsistencyLoss
from .config import Config

__all__ = [
    "TriBranchNetwork",
    "SelectiveConsistencyLoss",
    "Config",
]

