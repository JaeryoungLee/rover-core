SYSTEM=RoverBaseline   # RoverDark | RoverBaseline
# SYSTEM=RoverDark   # RoverDark | RoverBaseline

CONTROLLER=ParamMPC #MPC | MPC_NN
# CONTROLLER=MPC #MPC | MPC_NN
# CONTROLLER=NN #MPC | MPC_NN


# MPC

# === Stage 1: Build GridInput (MPC control evaluated on state grid) [~7 minutes]
python scripts/grid_input/build_grid_input.py --system $SYSTEM --input ${SYSTEM}_${CONTROLLER} --tag ${SYSTEM}_${CONTROLLER}
# python scripts/grid_input/visualize_grid_input.py --tag ${SYSTEM}_${CONTROLLER}

# === Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${SYSTEM}_${CONTROLLER} --tag ${SYSTEM}_${CONTROLLER}_Box
# python scripts/grid_set/visualize_grid_set.py --tag ${SYSTEM}_${CONTROLLER}_Box

# -------- NN Controller
# Train NN to approximate MPC (from GridInput cache) [~1.5 hours]
# python scripts/nn_input/train_nn_input.py --system $SYSTEM --tag ${SYSTEM}_${CONTROLLER} --hidden 128 --layers 2 --epochs 500000
# python scripts/nn_input/visualize_nn_input.py --tag ${SYSTEM}_${CONTROLLER}

# ------Sampling-based GridSet from NN (evaluate NN on grid, then sample uncertainty)
# python scripts/grid_input/build_grid_input.py --system $SYSTEM --nn-input-tag ${SYSTEM}_${CONTROLLER} --tag ${SYSTEM}_${CONTROLLER}_Grid
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${SYSTEM}_${CONTROLLER}_Grid --tag ${SYSTEM}_${CONTROLLER}_Box_Unclamped
# python scripts/grid_set/constrain_grid_set.py --grid-set-tag ${SYSTEM}_${CONTROLLER}_Box_Unclamped --tag ${SYSTEM}_${CONTROLLER}_Box --force
# python scripts/grid_set/visualize_grid_set.py --tag ${SYSTEM}_${CONTROLLER}_Box


# ====== Stage 3: Build GridValue (BRT for worst-case estimation error) [<1 minute]
# python scripts/grid_value/build_grid_value.py --dynamics $SYSTEM --control-grid-set-tag ${SYSTEM}_${CONTROLLER}_Box --tag ${SYSTEM}_${CONTROLLER}_WorstCase --force
# python scripts/grid_value/visualize_grid_value.py --tag ${SYSTEM}_${CONTROLLER}_WorstCase


# ===== Simulation 
# Simulate batch of challenging initial states under worst-case uncertainty [<1 minute]
# python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${SYSTEM}_${CONTROLLER}_sim --set uncertainty_tag=${SYSTEM}_${CONTROLLER}_WorstCase
# python scripts/simulation/visualize_simulation.py --tag ${SYSTEM}_${CONTROLLER}_sim --save-final-frame --value-tag ${SYSTEM}_${CONTROLLER}_WorstCase --value-time 0.0 --value-zero-level
# python scripts/simulation/visualize_simulation.py --tag ${SYSTEM}_${CONTROLLER}_sim --force
# python scripts/simulation/inspect_simulation.py --tag ${SYSTEM}_${CONTROLLER}_sim 






# Nominal

python scripts/grid_value/build_grid_value.py --dynamics ${SYSTEM}Nominal --control-grid-input-tag ${SYSTEM}_${CONTROLLER} --tag ${SYSTEM}_${CONTROLLER}_Nominal
python scripts/grid_value/visualize_grid_value.py --tag ${SYSTEM}_${CONTROLLER}_Nominal
python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${SYSTEM}_${CONTROLLER}_Nominal_sim --uncertainty-tag ${SYSTEM}_${CONTROLLER}_Nominal --set uncertainty=ZeroInput
python scripts/simulation/visualize_simulation.py --tag ${SYSTEM}_${CONTROLLER}_Nominal_sim --save-final-frame --value-tag ${SYSTEM}_${CONTROLLER}_Nominal --value-time 0.0 --value-zero-level
python scripts/simulation/visualize_simulation.py --tag ${SYSTEM}_${CONTROLLER}_Nominal_sim --force
python scripts/simulation/inspect_simulation.py --tag ${SYSTEM}_${CONTROLLER}_Nominal_sim 

