#!/bin/bash
# Run all unit tests. Run this BEFORE submitting any cluster job.
# Usage: bash tests/run_all.sh

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  test_attacks.py  (MIA attacks)"
echo "========================================"
python tests/test_attacks.py

echo ""
echo "========================================"
echo "  test_features.py  (DPRI features)"
echo "========================================"
python tests/test_features.py 2>&1 | grep -vE "\[DPRI\]"

echo ""
echo "========================================"
echo "  test_pipeline.py  (regression + CLI)"
echo "========================================"
python tests/test_pipeline.py 2>&1 | grep -E "PASS|FAIL|ERROR|passed"

echo ""
echo "========================================"
echo "  test_findings.py  (Finding 2)"
echo "========================================"
python tests/test_findings.py 2>&1 | grep -E "PASS|FAIL|ERROR|passed"

echo ""
echo "All test suites completed."
