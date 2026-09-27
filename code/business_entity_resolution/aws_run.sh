#!/usr/bin/env bash
# Full pipeline on a Linux box (e.g. AWS EC2). Usage:
#   bash aws_run.sh /path/to/student_resource [N_S1] [K] [LIMIT_S1]
# LIMIT_S1 (optional) = sanity run that only scores the first N test S1 rows.
# Recommended instance: >= 32 GB RAM, >= 8 vCPU (e.g. r6i.2xlarge / m6i.4xlarge).
set -euo pipefail
DATA=${1:?path to student_resource (contains dataset/ and utils/)}
N_S1=${2:-300000}
K=${3:-20}
LIMIT=${4:+--limit-s1 $4}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HERE/../../work}
OUT=${OUT:-$HERE/../../output}
export ER_WORKERS=${ER_WORKERS:-$(nproc)}

python3 -m venv "$HERE/.venv"
source "$HERE/.venv/bin/activate"
pip install -q -r "$HERE/requirements.txt"

cd "$HERE/src"
python run_pipeline.py --data-dir "$DATA" --work-dir "$WORK" --out-dir "$OUT" \
    --n-s1 "$N_S1" --k "$K" $LIMIT 2>&1 | tee "$HERE/pipeline.log"
