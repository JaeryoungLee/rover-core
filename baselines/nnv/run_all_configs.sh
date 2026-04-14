#!/bin/bash
# Run NNV baseline for multiple configurations SEQUENTIALLY.
# (Parallel MATLAB instances cause parpool conflicts.)
# Usage:
#   bash baselines/nnv/run_all_configs.sh                              # defaults
#   bash baselines/nnv/run_all_configs.sh T=5:ntheta=1 T=5:ntheta=100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NNV_DIR="$REPO_ROOT/libraries/NNV/code/nnv"

if [ $# -eq 0 ]; then
    CONFIGS="T=5:ntheta=1 T=5:ntheta=5 T=5:ntheta=10 T=5:ntheta=25 T=5:ntheta=50 T=5:ntheta=100"
else
    CONFIGS="$@"
fi

for CFG in $CONFIGS; do
    T_VAL=$(echo "$CFG" | grep -oP 'T=\K[0-9.]+')
    NT_VAL=$(echo "$CFG" | grep -oP 'ntheta=\K[0-9]+')
    T_VAL=${T_VAL:-5}
    NT_VAL=${NT_VAL:-100}

    LOGFILE="/tmp/nnv_T${T_VAL}_ntheta${NT_VAL}.log"
    echo "[$(date +%H:%M:%S)] Running T=${T_VAL}s ntheta=${NT_VAL} → ${LOGFILE}"
    matlab -batch "cd('$NNV_DIR'); startup_nnv; cd('$SCRIPT_DIR'); T_arg=$T_VAL; ntheta_arg=$NT_VAL; verify_reversed_dubins" \
        > "$LOGFILE" 2>&1
    EXIT=$?
    RUNTIME=$(grep -oP 'Runtime: \K[0-9.]+' "$LOGFILE" 2>/dev/null | head -1)
    echo "  → exit=$EXIT, runtime=${RUNTIME:-?}s"
done

echo ""
echo "Extracting all results ..."
cd "$REPO_ROOT"
source env/bin/activate 2>/dev/null || true
python baselines/nnv/extract_results.py

echo "Done."
