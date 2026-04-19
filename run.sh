SYSTEM=RoverDark   # RoverDark | RoverBaseline

# MPC

# === Stage 1: Build GridInput (MPC control evaluated on state grid) [~7 minutes]
# python scripts/grid_input/build_grid_input.py --system $SYSTEM --input ${SYSTEM}_MPC --tag ${SYSTEM}_MPC
# python scripts/grid_input/visualize_grid_input.py --tag ${SYSTEM}_MPC

# === Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${SYSTEM}_MPC --tag ${SYSTEM}_MPC_Box
# python scripts/grid_set/visualize_grid_set.py --tag ${SYSTEM}_MPC_Box

# === NN Controller
# Train NN to approximate MPC (from GridInput cache) [~1.5 hours]
# python scripts/nn_input/train_nn_input.py --system $SYSTEM --input-tag ${SYSTEM}_MPC --tag ${SYSTEM}_MPC_NN --hidden 128 --layers 2 --epochs 500000
# python scripts/nn_input/visualize_nn_input.py --tag ${SYSTEM}_MPC_NN 

# ------ Option A: Sampling-based GridSet from NN (evaluate NN on grid, then sample uncertainty)
# python scripts/grid_input/build_grid_input.py --system $SYSTEM --nn-input-tag ${SYSTEM}_MPC_NN --tag ${SYSTEM}_MPC_NN_Grid
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${SYSTEM}_MPC_NN_Grid --tag ${SYSTEM}_MPC_NN_Box
# python scripts/grid_set/constrain_grid_set.py --grid-set-tag ${SYSTEM}_MPC_NN_Box --tag ${SYSTEM}_MPC_NN_Box_Clamped
# python scripts/grid_set/visualize_grid_set.py --tag ${SYSTEM}_MPC_NN_Box_Clamped

# ------ Option B: Formal bounds via auto_LiRPA [~1.5 hours]
# python scripts/nn_input/build_grid_set.py --system $SYSTEM --nn-input-tag ${SYSTEM}_MPC_NN --tag ${SYSTEM}_MPC_NN_Box --method CROWN-IBP
# python scripts/grid_set/constrain_grid_set.py --grid-set-tag ${SYSTEM}_MPC_NN_Box --tag ${SYSTEM}_MPC_NN_Box_Clamped

# ====== Stage 3: Build GridValue (BRT for worst-case estimation error) [<1 minute]
# python scripts/grid_value/build_grid_value.py --dynamics RoverDark --control-grid-set-tag RoverDark_MPC_NN_Box_Clamped --tag RoverDark_WorstCase
# python scripts/grid_value/visualize_grid_value.py --tag RoverDark_WorstCase

# ===== Simulation 
# Simulate batch of challenging initial states under worst-case uncertainty [<1 minute]
# python scripts/simulation/simulate.py --system RoverDark --preset default --tag roverdark_sim
# python scripts/simulation/visualize_simulation.py --tag roverdark_sim
python scripts/simulation/inspect_simulation.py --tag roverdark_sim
