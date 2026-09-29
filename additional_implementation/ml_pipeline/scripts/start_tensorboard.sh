#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="${1:-runs}"
tensorboard --logdir "$RUN_ROOT" --bind_all