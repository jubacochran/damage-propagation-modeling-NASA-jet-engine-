from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class HPCStallMargin:
    # Constants and baseline values
    HPC_stall_margin_threshold = int
    eta_0 = int
    baseline_flow = int
