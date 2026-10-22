#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv-vision/bin/python ]]; then
  python3 -m venv .venv-vision
fi
if ! .venv-vision/bin/python -c 'import ultralytics' >/dev/null 2>&1; then
  .venv-vision/bin/python -m pip install -r requirements-vision.txt
fi
.venv-vision/bin/python train_lugnut.py "$@"
read '?Press Return to close. '
