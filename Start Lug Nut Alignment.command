#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv-vision/bin/python ]]; then
  python3 -m venv .venv-vision
fi
if ! .venv-vision/bin/python -c 'import ultralytics, fastapi, uvicorn' >/dev/null 2>&1; then
  .venv-vision/bin/python -m pip install -r requirements-vision.txt
fi
if curl -fsS http://127.0.0.1:8767/vision/status >/dev/null 2>&1; then
  open http://127.0.0.1:8767/align
  exit 0
fi
(sleep 2; open http://127.0.0.1:8767/align) &
exec .venv-vision/bin/python -m uvicorn live_app:app --host 127.0.0.1 --port 8767 --ws-max-size 4000000
