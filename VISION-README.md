# Wrist-camera lug-nut detection and image alignment

This extends the original manual tracker with a **trainable local YOLO detector**,
human-reviewed data capture, and image-space alignment guidance for a camera
rigidly mounted near an SO-ARM socket.

**Current delivery: the capture/training/inference software is implemented. No
real lug-nut dataset or trained lug-nut weights are included. No physical arm
alignment has been validated.** Generic pretrained COCO weights do not have a
lug-nut class; the live detector rejects them. The original `/` manual tracker
remains available.

## First use

1. Open **Start Lug Nut Alignment.command**. On the first run it installs the
   vision dependencies in `.venv-vision`. The app opens at
   <http://127.0.0.1:8767/align>.
2. Mount the camera rigidly near the socket. Start the camera and choose it from
   the device menu. Keep the arm stationary while collecting the first examples.
3. Under **Examples**, freeze a clear view. Draw a tight box around every visible
   lug nut. Label the nut, not its wheel recess or surrounding hole. Skip frames
   where you cannot confidently identify the visible nut boundary.
4. Confirm the labels and save. Resume the source to capture a different view.
   Photos and videos can also be opened locally. Reviewed frames without any lug
   nuts are useful negative examples; save them with zero boxes.
5. Use a session name for each recording. Reserve **separate recordings** for
   Validation and Final test. Never split nearby frames of the same video across
   training and validation. The app prevents reuse of a session name across
   splits and rejects exact duplicate frames; you still need to ensure that
   differently named sessions actually come from different recordings.
6. Open **Train Lug Nut Model.command**. This trains locally, downloads the
   initial YOLO11n weights on first use, and saves `models/lugnut.pt` plus
   `models/training-report.json`. Training can take substantial time on CPU.
7. Open **Alignment** and start detection on a live camera or video. Set the
   socket aim point, then click inside one detected lug nut to lock it.

The launcher is ready on this Mac. No cloud API key is required, and captured
images are not uploaded by this application. Data, model weights, and training
outputs are excluded from Git.

## Data and evaluation

A practical starting target is about **150–300 varied reviewed images**, plus
separate validation and test recordings. This is an engineering starting point,
not an accuracy guarantee. Include the actual camera, wheel and nuts, several
distances and angles, different lighting, chrome glare, shadows, occlusion by the
tool, and hard negatives such as valve stems, hub details and empty lug holes.
Avoid collecting hundreds of near-identical frames.

The training preflight requires at least 10 training and 5 validation images,
with positive examples in each used split. That is only a smoke-training floor,
not sufficient evidence for deployment. Images and YOLO-format labels are stored
in `data/lugnut/images/{train,val,test}` and `data/lugnut/labels/{train,val,test}`.
The manifest records recording sessions. `dataset.yaml` is generated on save.

Validation reports precision, recall, and detection mAP from Ultralytics. The
optional final test split is evaluated only after training. These metrics do not
measure socket alignment accuracy. Before robot integration, separately measure:

- Center error against manually marked nut centers on unseen camera frames.
- Missed detections, false detections and mistaken target associations.
- Latency, loss handling and center jitter at the actual working distances.
- Physical alignment residual and successful socket seating on supervised trials.

Keep final-test recordings out of model selection and tuning. An image detector
trained on one wheel is not established to work on different wheels or nut types.

## What the alignment output means

Coordinates are normalized internally; the displayed error is
`lug_box_center - socket_aim` in pixels. X points right and Y points down in the
camera image. These are **not robot axes or motor directions**.

The aim point starts unset. To calibrate it, first align the socket by hand at
the intended non-contact working distance and identify where its axis intersects
the lug face in the image. Set that image point. Do not simply assume the camera
image center is the socket axis. An offset camera introduces parallax, so a
different working distance can require a different aim point. Camera or source
changes clear the aim; remounting, rotating, changing the tool or working distance
requires setting it again even if image dimensions stay the same.

“Image centered” requires five consecutive fresh observations within the pixel
tolerance (default 5 px). This is a UI threshold, not a measured millimeter
accuracy. The detector score is not a calibrated probability of correct alignment.
The center of a bounding box is only an approximation of the nut axis, especially
for tilted views and partial occlusion.

Loss, ambiguity, resolution changes, and gaps over 750 ms clear the target.
The browser clears guidance on slow round trips, paused/ended video, hidden tabs,
camera disconnection, and connection failure. It sends at most one frame at a time
and does not use repeated still-photo frames to claim stable alignment. Slow first
inference can be discarded while the model loads; later frames can be selected.
Target association uses nearby centers and box size. It can still be fooled by
nearby lookalike objects or an incorrect detector: it is not a robot safety system.

No robot driver, serial connection, motion endpoint or tool-rotation command is
included. The next integration needs a measured image-to-motion relationship
(visual-servo Jacobian or calibrated camera/robot geometry), joint/workspace limits,
hold-to-run control, and independent depth, orientation and contact checks. 2D
centering alone cannot establish those quantities.

## Developer commands

```sh
# Base tests: no model download or hardware access
.venv/bin/python -m pytest -q
node --test tests/*.test.cjs
node --check alignment.js

# Optional vision environment
python3 -m venv .venv-vision
.venv-vision/bin/python -m pip install -r requirements-vision.txt
.venv-vision/bin/python -m uvicorn live_app:app --host 127.0.0.1 --port 8767 --ws-max-size 4000000

# Preflight / training (use --device mps for an Apple GPU when supported)
.venv-vision/bin/python train_lugnut.py --check-only
.venv-vision/bin/python train_lugnut.py --epochs 60 --device cpu
```

`GET /vision/status` reports model availability and dataset counts.
`POST /vision/examples` saves reviewed labels and a frame, with same-origin checks.
`WS /vision/detect` accepts `configure`, `frame`, `select` and `clear` messages.
Results include normalized detections, image offsets, freshness state and the
constant `robot_motion_enabled: false`. See `tests/test_vision.py` for the protocol.
Keep the service bound to localhost. Only load trusted model checkpoints that you
trained or obtained from a trusted source.

Development checks passed on Python 3.12, Ultralytics 8.4.184 and PyTorch 2.14.1
on this Mac. A temporary one-epoch synthetic training/inference smoke run verified
that the actual ML pipeline executes; those weights were discarded. It provides
no evidence of recognition accuracy on real lug nuts. Browser camera permission,
physical wrist-camera capture and robot alignment still need hardware verification.

## Dependencies and references

The optional training and inference implementation uses Ultralytics YOLO and its
upstream license terms. See the [Ultralytics repository and license](https://github.com/ultralytics/ultralytics),
[custom detection dataset format](https://docs.ultralytics.com/datasets/detect/),
and [prediction API](https://docs.ultralytics.com/modes/predict/).
For later camera/robot calibration, see
[OpenCV camera calibration and hand-eye calibration](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html).
