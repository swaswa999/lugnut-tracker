/* Local wrist-camera dataset and inference client. No robot endpoints. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('view'), ctx = canvas.getContext('2d'), video = $('source');
  const frameCanvas = document.createElement('canvas'), frameCtx = frameCanvas.getContext('2d');
  let mode = 'label', stream = null, image = null, objectURL = null, sourceEpoch = 0;
  let frozen = false, frozenData = null, boxes = [], drag = null, aim = null, settingAim = false;
  let socket = null, running = false, configured = false, pending = null, frameID = 0, result = null;
  let lastResultAt = 0, lastMediaTime = -1, lastSentAt = 0, weightsAvailable = false;
  const say = text => { $('message').textContent = text; };
  const stateNames = {select_target:'Select a detected lug nut',set_aim:'Set the socket aim point',
    acquiring:'Acquiring target',offset:'Target offset',image_aligned:'Image centered',
    stale:'Frame too old — reselect target',target_lost:'Target lost — reselect',
    ambiguous:'Ambiguous target — reselect',camera_changed:'View changed — set aim again',error:'Detection error'};

  function invalidate(text='No fresh measurement') {
    result = null; lastResultAt = 0;
    $('error').textContent = '—'; $('state').textContent = text; $('metrics').textContent = 'No fresh measurement';
  }
  function stopDetection() {
    running = false; configured = false; pending = null;
    if (socket) { const old = socket; socket = null; old.close(); }
    $('detect').textContent = 'Start detection'; invalidate();
  }
  function stopSource() {
    sourceEpoch++; stopDetection();
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; video.pause(); video.srcObject = null; video.removeAttribute('src'); video.load();
    if (objectURL) URL.revokeObjectURL(objectURL);
    objectURL = null; image = null; frozen = false; frozenData = null; boxes = []; aim = null;
    settingAim = false; $('aim').textContent = 'Set socket aim point';
    lastMediaTime = -1; $('empty').hidden = false; $('sourceName').textContent = 'No source selected.';
    $('reviewed').checked = false; updateBoxes();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  function sourceReady() { return !!image || (video.readyState >= 2 && video.videoWidth > 0); }
  function sizeToSource() {
    const w = image ? image.naturalWidth : video.videoWidth;
    const h = image ? image.naturalHeight : video.videoHeight;
    if (!w || !h) return false;
    const factor = Math.min(1, 1280 / Math.max(w, h));
    const width = Math.round(w * factor), height = Math.round(h * factor);
    if (canvas.width !== width || canvas.height !== height || frameCanvas.width !== width || frameCanvas.height !== height) {
      canvas.width = frameCanvas.width = width; canvas.height = frameCanvas.height = height;
      aim = null; boxes = []; invalidate('View changed — set aim again');
    }
    return true;
  }
  function capture() {
    if (!sourceReady() || !sizeToSource()) throw Error('Start a camera or open a photo/video first.');
    frameCtx.drawImage(image || video, 0, 0, frameCanvas.width, frameCanvas.height);
    return frameCanvas.toDataURL('image/jpeg', .9).split(',')[1];
  }
  async function cameras() {
    if (!navigator.mediaDevices) throw Error('Open the app at localhost to access cameras.');
    const previous = $('cameras').value;
    const list = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    $('cameras').replaceChildren(new Option('Default camera', ''));
    list.forEach((d, i) => $('cameras').add(new Option(d.label || `Camera ${i+1}`, d.deviceId)));
    if (list.some(d => d.deviceId === previous)) $('cameras').value = previous;
  }
  async function startCamera() {
    stopSource(); const epoch = sourceEpoch;
    say('Waiting for camera access…');
    const deviceId = $('cameras').value;
    const next = await navigator.mediaDevices.getUserMedia({audio:false,video:{
      width:{ideal:1280},height:{ideal:720},...(deviceId ? {deviceId:{exact:deviceId}} : {})}});
    if (epoch !== sourceEpoch) { next.getTracks().forEach(t => t.stop()); return; }
    stream = next; video.srcObject = stream;
    stream.getVideoTracks()[0].onended = () => { stopSource(); say('Camera disconnected. Reconnect it and start again.'); };
    await video.play(); await cameras();
    $('empty').hidden = true; $('sourceName').textContent = 'Wrist camera · live'; say('');
  }
  async function openFile(file) {
    if (!file) return;
    stopSource(); const epoch = sourceEpoch; objectURL = URL.createObjectURL(file);
    if (file.type.startsWith('image/')) {
      const next = new Image(); next.src = objectURL; await next.decode();
      if (epoch !== sourceEpoch) return; image = next;
    } else {
      video.src = objectURL; video.loop = false; await video.play();
      if (epoch !== sourceEpoch) return;
    }
    $('empty').hidden = true; $('sourceName').textContent = file.name; say('');
  }
  function updateBoxes() { $('boxCount').textContent = `${boxes.length} label${boxes.length === 1 ? '' : 's'}`; }
  function freeze() {
    stopDetection(); frozenData = capture(); frozen = true; video.pause();
    boxes = []; $('reviewed').checked = false; updateBoxes(); say('Draw a tight box around every visible lug nut.');
  }
  async function resume() {
    frozen = false; frozenData = null; boxes = []; $('reviewed').checked = false; updateBoxes();
    if (!image && sourceReady()) await video.play(); say('');
  }
  async function refreshStatus() {
    const response = await fetch('/vision/status');
    if (!response.ok) throw Error('Could not read model status.');
    const status = await response.json();
    for (const split of ['train','val','test']) $(split+'Count').textContent = status.dataset[split];
    weightsAvailable = status.detector.weights_available;
    $('modelStatus').textContent = weightsAvailable ? 'Local model available. Verify detections on new footage.' : 'No trained model yet. Capture examples, then train.';
    $('detect').disabled = !weightsAvailable;
  }
  async function save() {
    if (!frozen || !frozenData) throw Error('Freeze a frame before saving.');
    if (!$('reviewed').checked) throw Error('Review all labels and check the confirmation box.');
    $('save').disabled = true;
    try {
      const response = await fetch('/vision/examples', {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({frame:frozenData,boxes,session:$('session').value,split:$('split').value,reviewed:true})});
      const body = await response.json(); if (!response.ok) throw Error(body.detail || 'Save failed.');
      say('Saved. Resume and capture a different view.'); await refreshStatus();
    } finally { $('save').disabled = false; }
  }
  function configure() {
    configured = false; pending = null;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type:'configure',aim,
      tolerance_px:Number($('tolerance').value),confidence:Number($('confidence').value)}));
    invalidate();
  }
  async function startDetection() {
    if (running) { stopDetection(); return; }
    await refreshStatus(); if (!weightsAvailable) throw Error('Capture and train a model first.');
    if (!sourceReady()) throw Error('Start a camera or open a video first.');
    if (image) throw Error('Use live camera or video for alignment; a still image cannot establish stable alignment.');
    await resume(); running = true; lastMediaTime = -1;
    const ws = new WebSocket(`ws://${location.host}/vision/detect`); socket = ws;
    $('detect').textContent = 'Stop detection';
    ws.onopen = () => { if (socket === ws) configure(); };
    ws.onmessage = event => {
      if (socket !== ws) return;
      const data = JSON.parse(event.data);
      if (data.type === 'error') { pending = null; say(data.message); stopDetection(); return; }
      if (data.type === 'configured') { configured = true; return; }
      if (data.type !== 'result') return;
      if (!pending || pending.id !== data.frame_id) return;
      const elapsed = performance.now() - pending.sent; pending = null;
      if (elapsed > 750 || video.paused || video.ended || document.hidden) {
        invalidate('Frame too old — no guidance'); ws.send(JSON.stringify({type:'clear'})); return;
      }
      result = data; lastResultAt = performance.now();
      const a = data.alignment; $('state').textContent = stateNames[a.state] || a.state;
      $('error').textContent = a.error_px ? `${a.error_px[0].toFixed(1)}, ${a.error_px[1].toFixed(1)}` : '—';
      $('metrics').textContent = `X, Y in pixels · ${data.processing_ms} ms · ${a.stable_frames}/5 fresh frames`;
    };
    ws.onclose = () => { if (socket === ws) { stopDetection(); say('Detection connection closed.'); } };
    ws.onerror = () => { if (socket === ws) { stopDetection(); say('Could not connect to the local detector.'); } };
  }
  function drawBox(box, color, label) {
    const [x1,y1,x2,y2] = box, w=canvas.width,h=canvas.height;
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(x1*w,y1*h,(x2-x1)*w,(y2-y1)*h);
    ctx.font='14px system-ui';ctx.fillStyle=color;ctx.fillText(label,x1*w+3,Math.max(16,y1*h-5));
  }
  function draw() {
    const now = performance.now();
    if (sourceReady()) {
      if (!frozen) { sizeToSource(); frameCtx.drawImage(image || video,0,0,frameCanvas.width,frameCanvas.height); }
      ctx.drawImage(frameCanvas,0,0);
      if (mode === 'label') {
        boxes.forEach((b,i) => drawBox(b,'#bdf477',`lug ${i+1}`));
        if (drag) drawBox(normalize(drag.start,drag.end),'#f1c776','');
      } else {
        if (aim) {
          const x=aim[0]*canvas.width,y=aim[1]*canvas.height;
          ctx.strokeStyle='#f1c776';ctx.lineWidth=2;ctx.beginPath();
          ctx.moveTo(x-14,y);ctx.lineTo(x+14,y);ctx.moveTo(x,y-14);ctx.lineTo(x,y+14);ctx.stroke();
        }
        if (result && now-lastResultAt <= 750) {
          result.detections.forEach(d => drawBox(d.xyxy,'#a9c4ff',`${Math.round(d.confidence*100)}% score`));
          const a=result.alignment;
          if (a.target) {
            drawBox(a.target.xyxy,a.image_aligned?'#bdf477':'#f1c776','TARGET');
            if (aim) { const b=a.target.xyxy;ctx.beginPath();ctx.moveTo(aim[0]*canvas.width,aim[1]*canvas.height);
              ctx.lineTo((b[0]+b[2])/2*canvas.width,(b[1]+b[3])/2*canvas.height);ctx.stroke(); }
          }
        }
      }
    }
    if (result && (now-lastResultAt > 750 || video.ended || video.paused || document.hidden)) {
      invalidate('No fresh frame — reselect target');
      if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type:'clear'}));
    }
    if (pending && now-pending.sent > 5000) { stopDetection(); say('Detector timed out. Check the server and restart detection.'); }
    if (running && configured && socket?.readyState === WebSocket.OPEN && !pending && !frozen && !document.hidden
        && sourceReady() && !video.paused && !video.ended && video.currentTime !== lastMediaTime && now-lastSentAt >= 100) {
      const id=++frameID;pending={id,sent:now};lastMediaTime=video.currentTime;lastSentAt=now;
      socket.send(JSON.stringify({type:'frame',frame_id:id,frame:capture()}));
    }
    requestAnimationFrame(draw);
  }
  const normalizedPoint = event => {
    const rect=canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
  };
  const normalize = (a,b) => [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];
  canvas.onpointerdown = event => {
    const p=normalizedPoint(event);
    if (mode==='label' && frozen) {drag={start:p,end:p};canvas.setPointerCapture(event.pointerId);}
    else if (mode==='align' && settingAim && sourceReady()) {
      aim=p;settingAim=false;$('aim').textContent='Set socket aim point';configure();say('Aim set for this camera view and working distance.');
    } else if (mode==='align' && result && socket?.readyState===WebSocket.OPEN && performance.now()-lastResultAt<750) {
      socket.send(JSON.stringify({type:'select',frame_id:result.frame_id,point:p}));
    } else if (mode==='align' && running) say('Wait for a fresh detection, then click the lug nut.');
  };
  canvas.onpointermove = event => { if(drag) drag.end=normalizedPoint(event); };
  canvas.onpointerup = event => {
    if (!drag) return;const box=normalize(drag.start,normalizedPoint(event));drag=null;
    if ((box[2]-box[0])*canvas.width>=6 && (box[3]-box[1])*canvas.height>=6 && boxes.length<20) boxes.push(box);
    $('reviewed').checked=false;updateBoxes();
  };
  canvas.onpointercancel = () => {drag=null;};
  function setMode(next) {
    stopDetection();mode=next;settingAim=false;drag=null;
    $('labelPanel').hidden=next!=='label';$('alignPanel').hidden=next!=='align';
    $('labelTab').setAttribute('aria-selected',String(next==='label'));
    $('alignTab').setAttribute('aria-selected',String(next==='align'));
    refreshStatus().catch(e=>say(e.message));
  }
  function action(id, fn) { $(id).onclick=()=>Promise.resolve().then(fn).catch(e=>say(e.message)); }
  action('camera',startCamera);action('refresh',cameras);action('stop',stopSource);
  action('freeze',freeze);action('resume',resume);action('save',save);action('detect',startDetection);
  action('undo',()=>{boxes.pop();$('reviewed').checked=false;updateBoxes();});
  action('clearBoxes',()=>{boxes=[];$('reviewed').checked=false;updateBoxes();});
  action('labelTab',()=>setMode('label'));action('alignTab',()=>setMode('align'));
  action('aim',()=>{settingAim=true;say('Click the image location corresponding to the socket axis at your working distance.');$('aim').textContent='Click aim point in image…';});
  action('resetTarget',()=>{if(socket?.readyState===WebSocket.OPEN)socket.send(JSON.stringify({type:'clear'}));invalidate();});
  $('confidence').oninput=()=>{$('confidenceValue').textContent=Number($('confidence').value).toFixed(2);};
  $('confidence').onchange=configure;$('tolerance').onchange=configure;
  $('file').onchange=event=>openFile(event.target.files[0]).catch(e=>say(e.message));
  $('cameras').onchange=()=>{if(stream)startCamera().catch(e=>say(e.message));};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopDetection();});
  video.addEventListener('ended',()=>{stopDetection();say('Video ended. Replay or select another source.');});
  window.addEventListener('beforeunload',stopSource);
  if(navigator.mediaDevices) navigator.mediaDevices.addEventListener('devicechange',()=>cameras().catch(e=>say(e.message)));
  refreshStatus().catch(e=>say(e.message));cameras().catch(()=>{});requestAnimationFrame(draw);
})();
