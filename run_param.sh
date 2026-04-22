#!/usr/bin/env bash
set -e

# Activate the project venv so hj_reachability / casadi / etc. resolve correctly,
# regardless of how the script is launched.
source "$(dirname "$0")/env/bin/activate"

EXP=sparse # sparse | medium | dense

SYSTEM=RoverParam   # RoverDark | RoverBaseline | RoverParam
# SYSTEM=RoverDark   # RoverDark | RoverBaseline

# CONTROLLER=ParamMPC #MPC | MPC_NN
CONTROLLER=MPC #MPC | MPC_NN
# CONTROLLER=NN #MPC | MPC_NN

# UNC=small # zero | small | moderate | harsh



# MPC
python scripts/param_mpc/visualize_param_mpc.py --lambdas 0.2 0.4 0.6 0.8 1.0 

# === Stage 1: Build GridInput (MPC control evaluated on state grid) [~7 minutes]

# for EXP in sparse medium dense; do
#     export ROVER_PARAM_SCENARIO=$EXP
#     python scripts/grid_input/build_grid_input.py --system $SYSTEM --input ${SYSTEM}_${CONTROLLER} --tag ${EXP}_${SYSTEM}_${CONTROLLER} --force
#     python scripts/grid_input/visualize_grid_input.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}
# done


# for EXP in sparse medium dense; do
#     # Link tag prefix to the obstacle scenario picked up by RoverParam at import time
    # export ROVER_PARAM_SCENARIO=$EXP
    # for UNC in zero small moderate harsh; do
#         # Uncertainty
#         # === Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
#         python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${EXP}_${SYSTEM}_${CONTROLLER} --tag ${EXP}_${SYSTEM}_${CONTROLLER}_Box_${UNC} --uncertainty-preset ${UNC} --force
#         python scripts/grid_set/visualize_grid_set.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_Box_${UNC}


#         # ====== Stage 3: Build GridValue (BRT for worst-case estimation error) [<1 minute]
#         python scripts/grid_value/build_grid_value.py --dynamics ${SYSTEM} --reach-avoid --control-grid-set-tag ${EXP}_${SYSTEM}_${CONTROLLER}_Box_${UNC} --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} --force
#         python scripts/grid_value/visualize_grid_value.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}



#         # ====== Stage 4: Simulation
        # Override `control_tag` so the sim runs the same scenario-specific MPC grid the BRT was built from.
        # if [ "${UNC}" = "zero" ]; then
        #     python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim \
        #         --set control_tag=${EXP}_${SYSTEM}_${CONTROLLER} \
        #         --set uncertainty=ZeroInput \
        #         --sweep-dim 3 --sweep-values 0.2 0.4 0.6 0.8 1.0
        # else
        #     python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim \
        #         --set control_tag=${EXP}_${SYSTEM}_${CONTROLLER} \
        #         --set uncertainty_tag=${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} \
        #         --sweep-dim 3 --sweep-values 0.2 0.4 0.6 0.8 1.0
        # fi
        # python scripts/simulation/visualize_simulation.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim --reach-avoid --save-final-frame --value-tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} --value-time 0.0 --value-zero-level
        # # python scripts/simulation/visualize_simulation.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim --reach-avoid --value-tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} --value-time 0.0  --force
        # python scripts/simulation/inspect_simulation.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim 

#         # # ====== Stage 5: Find the optimal parameter
#         python scripts/param_mpc/find_best_lambda.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}

    # done
    # python scripts/param_mpc/visualize_best_lambda.py --tags \
    # ${EXP}_${SYSTEM}_${CONTROLLER}_zero ${EXP}_${SYSTEM}_${CONTROLLER}_small \
    # ${EXP}_${SYSTEM}_${CONTROLLER}_moderate ${EXP}_${SYSTEM}_${CONTROLLER}_harsh

# done



# python scripts/param_mpc/visualize_best_lambda_summary.py


