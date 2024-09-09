from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class HPCStallMargin:
    # Constants and baseline values
    HPC_stall_margin_threshold: float = 0.15
    eta_0: float = 0.40
    baseline_flow: float = +0.08
    T_std: float = 518.67
    P_std: float = 14.70
    
    #Stall Margin Coefficients
    a: float = -3.81
    b: float = +2.57
    c: float = 1.0

    #Wear Manifest ranges
    Baseline_Efficiency: float = 0.0
    Initial_Wear_HPT_Efficiency: float = -0.48
    Initial_Wear_HPT_Flow: float = +0.08
    Wear_HPT_Efficiency_3000: float = -2.63
    Wear_HPT_Flow_3000: float = +1.76
    Wear_HPT_Efficiency_6000: float = -3.81
    Wear_HPT_Flow_6000: float = +2.57


    #Compression process temp variable
    gamma: float = 1.4

    stall_margin_failure_threshold: float = None

    def calculate_health_metrics(self, data: pd.DataFrame) -> pd.DataFrame:


        #terms for formulas
        T_inlet = data['Fan inlet temperature ◦R']
        P_inlet = data['Fan inlet Pressure psia']
        T_outlet_actual = data['HPC outlet temperature ◦R']
        P_outlet = data['HPC outlet pressure psia']

        #Pressure ratio across HPC
        PR = P_outlet / P_inlet

        # Calculate ideal outlet temperature for isentropic process
        T_outlet_ideal = T_inlet * (PR ** ((self.gamma - 1) / self.gamma))

        # Current efficiency calculation
        eta = (T_outlet_ideal - T_inlet) / (T_outlet_actual - T_inlet)
        
        # Efficiency loss calculation
        efficiency_loss = self.eta_0 - eta
        data['efficiency_loss'] = efficiency_loss
        
        # Corrected flow calculation
        corrected_flow = np.sqrt(T_inlet / self.T_std) * (self.P_std / P_inlet)
        data['flow_loss'] = self.baseline_flow - corrected_flow

        # Stall margin calculation
        data['stall_margin'] = self.a * efficiency_loss + self.b * data['flow_loss'] + self.c
        
        # Define the nominal stall margin for comparison (for each engine separately)
        nominal_stall_margin = data['stall_margin'].max()

        # Calculate the threshold for failure
        self.stall_margin_failure_threshold = nominal_stall_margin * (1 - self.HPC_stall_margin_threshold)

        # Check if stall margin falls below the threshold and flag it
        data['HPC_failure'] = data['stall_margin'] < self.stall_margin_failure_threshold
        
        return data





