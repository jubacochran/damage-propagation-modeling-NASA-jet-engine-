from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class HPCStallMargin:
    # Constants and baseline values
    HPC_stall_margin_threshold: int = 0.15
    eta_0: int = 0.040
    baseline_flow: int = +0.08
    T_std: int = 518.67
    P_std: int = 14.70
    
    #Stall Margin Coefficients
    a: int = -3.81
    b: int = 2.57
    c: int = 1.0

    #Compression process temp variable
    gamma: float = 1.4

    def calculate_health_metricsf(self, data: pd.DataFrame) -> pd.DataFrame:

