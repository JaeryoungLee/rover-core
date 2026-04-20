SYSTEM=RoverBaseline   # RoverDark | RoverBaseline
# SYSTEM=RoverDark   # RoverDark | RoverBaseline

# CONTROLLER=ParamMPC #MPC | MPC_NN
CONTROLLER=MPC #MPC | MPC_NN
# CONTROLLER=NN #MPC | MPC_NN

UNC=zero  # zero | small | moderate | harsh



# MPC

# === Stage 1: Build GridInput (MPC control evaluated on state grid) [~7 minutes]
# python scripts/grid_input/build_grid_input.py --system $SYSTEM --input ${SYSTEM}_${CONTROLLER} --tag ${SYSTEM}_${CONTROLLER}
# python scripts/grid_input/visualize_grid_input.py --tag ${SYSTEM}_${CONTROLLER}

# === Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${SYSTEM}_${CONTROLLER} --tag ${SYSTEM}_${CONTROLLER}_Box
# python scripts/grid_set/visualize_grid_set.py --tag ${SYSTEM}_${CONTROLLER}_Box



# Uncertainty
# === Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
# python scripts/grid_set/build_grid_set.py --system $SYSTEM --grid-input-tag ${SYSTEM}_${CONTROLLER} --tag ${SYSTEM}_${CONTROLLER}_Box_${UNC} --uncertainty-preset ${UNC} --force
# python scripts/grid_set/visualize_grid_set.py --tag ${SYSTEM}_${CONTROLLER}_Box_${UNC}


# ====== Stage 3: Build GridValue (BRT for worst-case estimation error) [<1 minute]
# python scripts/grid_value/build_grid_value.py --dynamics ${SYSTEM}ReachAvoid --control-grid-set-tag ${SYSTEM}_${CONTROLLER}_Box_${UNC} --reach-avoid --tag ${SYSTEM}_${CONTROLLER}_RA_${UNC} --force
# python scripts/grid_value/visualize_grid_value.py --tag ${SYSTEM}_${CONTROLLER}_RA_${UNC}


if [ "${UNC}" = "zero" ]; then
    python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${SYSTEM}_${CONTROLLER}_RA_${UNC}_sim --set uncertainty=ZeroInput
else
    python scripts/simulation/simulate.py --system ${SYSTEM} --preset default --tag ${SYSTEM}_${CONTROLLER}_RA_${UNC}_sim --set uncertainty_tag=${SYSTEM}_${CONTROLLER}_RA_${UNC}
fi
python scripts/simulation/visualize_simulation.py --tag ${SYSTEM}_${CONTROLLER}_RA_${UNC}_sim --reach-avoid --save-final-frame --value-tag ${SYSTEM}_${CONTROLLER}_RA_${UNC} --value-time 0.0 --value-zero-level
python scripts/simulation/visualize_simulation.py --tag ${SYSTEM}_${CONTROLLER}_RA_${UNC}_sim --value-tag ${SYSTEM}_${CONTROLLER}_RA_${UNC} --value-time 0.0 --reach-avoid --force
python scripts/simulation/inspect_simulation.py --tag ${SYSTEM}_${CONTROLLER}_RA_${UNC}_sim 





