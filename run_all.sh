#!/usr/bin/env bash
set -euo pipefail

PY="E:/2/py/python.exe"

"$PY" -m src.experiments --datasets turtle lung diaphragm --output results/metrics.csv
"$PY" -m src.visualize
