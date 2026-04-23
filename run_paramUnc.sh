#!/usr/bin/env bash
set -e
ROOT="$(dirname "$(realpath "$0")")"
PYTHON="$ROOT/env/bin/python"
export PYTHONPATH="$ROOT/libraries/hj_reachability:${PYTHONPATH:-}"

EXP=obs_unc
export ROVER_PARAM_SCENARIO=sparse # sparse | medium | dense

SYSTEM=RoverParam   # RoverDark | RoverBaseline | RoverParam
# SYSTEM=RoverDark   # RoverDark | RoverBaseline

# CONTROLLER=ParamMPC #MPC | MPC_NN
CONTROLLER=MPC #MPC | MPC_NN
# CONTROLLER=NN #MPC | MPC_NN


# # === Stage 1: Build GridInput (MPC control evaluated on state grid) [~7 minutes]
# python scripts/grid_input/build_grid_input.py --system $SYSTEM --input ${SYSTEM}_${CONTROLLER} --tag ${EXP}_${CONTROLLER} --force
# python scripts/grid_input/visualize_grid_input.py --tag ${EXP}_${CONTROLLER}


# # # === Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${EXP}_${CONTROLLER} --tag ${EXP}_${CONTROLLER}_Box --force
# python scripts/grid_set/visualize_grid_set.py --tag ${EXP}_${CONTROLLER}_Box


# # # ====== Stage 3: Build GridValue (BRT for worst-case estimation error) [<1 minute]
# python scripts/grid_value/build_grid_value.py --dynamics ${SYSTEM} --reach-avoid --control-grid-set-tag ${EXP}_${CONTROLLER}_Box --tag ${EXP}_${CONTROLLER} --force
# python scripts/grid_value/visualize_grid_value.py --tag ${EXP}_${CONTROLLER}


python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${EXP}_${CONTROLLER}_sim \
    --set control_tag=${EXP}_${CONTROLLER} \
    --set uncertainty_tag=${EXP}_${CONTROLLER} \
    --sweep-dim 4 --sweep-values  0.7 0.3
    # --sweep-dim 3 --sweep-values 0.1 1.0 
    # 
    
            

python scripts/simulation/visualize_simulation.py --tag ${EXP}_${CONTROLLER}_sim  --reach-avoid --save-final-frame --value-tag ${EXP}_${CONTROLLER} --value-time 0.0 --value-zero-level   
python scripts/simulation/visualize_simulation.py --tag ${EXP}_${CONTROLLER}_sim  --reach-avoid  --value-tag ${EXP}_${CONTROLLER} --value-time 0.0 --value-zero-level --force
    



# python scripts/param_mpc/find_best_params_2d.py \
#     --tag ${EXP}_${CONTROLLER} \
#     --param-dim-a 3 --param-dim-b 4

# python scripts/param_mpc/visualize_best_params_2d.py --tag ${EXP}_${CONTROLLER}


# # for UNC in zero small moderate harsh; do
# for UNC in harsh; do
#     # # Uncertainty
#     # # === Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
#     # python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${EXP}_${SYSTEM}_${CONTROLLER} --tag ${EXP}_${SYSTEM}_${CONTROLLER}_Box_${UNC} --uncertainty-preset ${UNC} --force
#     # python scripts/grid_set/visualize_grid_set.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_Box_${UNC}


#     # # # ====== Stage 3: Build GridValue (BRT for worst-case estimation error) [<1 minute]
#     # python scripts/grid_value/build_grid_value.py --dynamics ${SYSTEM} --reach-avoid --control-grid-set-tag ${EXP}_${SYSTEM}_${CONTROLLER}_Box_${UNC} --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} --force
#     # python scripts/grid_value/visualize_grid_value.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}



#     # # # ====== Stage 4: Simulation
#     # # Override `control_tag` so the sim runs the same scenario-specific MPC grid the BRT was built from.
#     if [ "${UNC}" = "zero" ]; then
#         python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim \
#             --set control_tag=${EXP}_${SYSTEM}_${CONTROLLER} \
#             --set uncertainty=ZeroInput \
#             --sweep-dim 3 --sweep-values 0.0 1.0 
#             # --sweep-dim 4 --sweep-values 7.0 3.0
            

#     else
#         python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim \
#             --set control_tag=${EXP}_${SYSTEM}_${CONTROLLER} \
#             --set uncertainty_tag=${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} \
#             --sweep-dim 4 --sweep-values 7.0 3.0
#             # --sweep-dim 3 --sweep-values 0.1 1.0 
            

#     fi
#     python scripts/simulation/visualize_simulation.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim --reach-avoid --save-final-frame --value-tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} --value-time 0.0 --value-zero-level
#     # # python scripts/simulation/visualize_simulation.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim --reach-avoid --value-tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} --value-time 0.0  --force
#     # python scripts/simulation/inspect_simulation.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}_sim 

#     # # # ====== Stage 5: Find the optimal parameter
#     # # 2D parameter search over (λ, v) — dim 3 = λ, dim 4 = v
#     # python scripts/param_mpc/find_best_params_2d.py \
#     #     --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC} \
#     #     --param-dim-a 3 --param-dim-b 4

#     # python scripts/param_mpc/visualize_best_params_2d.py --tag ${EXP}_${SYSTEM}_${CONTROLLER}_${UNC}

# done

# # # Cross-uncertainty 2D heatmaps (one panel per uncertainty, shared color scale)
# # python scripts/param_mpc/visualize_best_params_2d.py --tags \
# #     ${EXP}_${SYSTEM}_${CONTROLLER}_zero ${EXP}_${SYSTEM}_${CONTROLLER}_small \
# #     ${EXP}_${SYSTEM}_${CONTROLLER}_moderate ${EXP}_${SYSTEM}_${CONTROLLER}_harsh \
# #     --shared-colorbar


# # python scripts/param_mpc/visualize_best_lambda_summary.py




