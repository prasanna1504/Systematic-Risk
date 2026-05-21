import sys
import os
import torch
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Add project root to path
sys.path.append(os.path.abspath('.'))

try:
    from models.rasr_ge import RASR_GE
    print("✓ Successfully imported RASR_GE")
    
    from stress.shock_engine import ShockEngine
    print("✓ Successfully imported ShockEngine")
    
    from stress.counterfactual import compute_regime_contagion_multiplier
    print("✓ Successfully imported compute_regime_contagion_multiplier")
    
    from risk.hmm_regime import RegimeDetector
    print("✓ Successfully imported RegimeDetector")
    
    from risk.var_engine import VaREngine
    print("✓ Successfully imported VaREngine")
    
    from risk.cva_engine import CVAEngine
    print("✓ Successfully imported CVAEngine")
    
    # Try a simple GATConv forward pass if possible
    from torch_geometric.nn import GATConv
    gat = GATConv(64, 32, heads=4, edge_dim=1)
    print("✓ Successfully instantiated GATConv")
    
except Exception as e:
    print(f"✗ Import/Init failed: {e}")
    import traceback
    traceback.print_exc()
