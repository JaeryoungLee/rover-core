SYSTEM=RoverDark   # RoverDark | RoverBaseline

# MPC

# Stage 1: Build GridInput (MPC control evaluated on state grid) [~7 minutes]
# python scripts/grid_input/build_grid_input.py --system $SYSTEM --input ${SYSTEM}_MPC --tag ${SYSTEM}_MPC
# python scripts/grid_input/visualize_grid_input.py --tag ${SYSTEM}_MPC

# Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${SYSTEM}_MPC --tag ${SYSTEM}_MPC_Box


# NN Controller
# Train NN to approximate MPC (from GridInput cache) [~1.5 hours]
# python scripts/nn_input/train_nn_input.py --system $SYSTEM --input-tag ${SYSTEM}_MPC --tag ${SYSTEM}_MPC_NN --hidden 128 --layers 2 --epochs 500000
python scripts/nn_input/visualize_nn_input.py --tag ${SYSTEM}_MPC_NN 
# Compute NN output bounds via auto_LiRPA, then clamp to control limits [~1.5 hours]
# python scripts/nn_input/build_grid_set.py --system $SYSTEM --nn-input-tag ${SYSTEM}_MPC_NN --tag ${SYSTEM}_MPC_NN_Box --method CROWN-IBP
# will this work?
# python build_grid_input.py --system $SYSTEM --input ${SYSTEM}_NN --tag ${SYSTEM}_MPC_NN_Box


# python scripts/grid_set/constrain_grid_set.py --grid-set-tag ${SYSTEM}_MPC_NN_Box --tag ${SYSTEM}_MPC_NN_Box_Clamped

