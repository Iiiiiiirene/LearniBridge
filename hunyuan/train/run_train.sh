#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/scripts/train_schedule.py" --model hunyuan "$@"
