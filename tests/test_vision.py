import base64
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from alignment import Alignment
from live_app import app
from train_lugnut import check_dataset
from vision_data import Dataset
from vision_model import LugnutDetector
import vision_routes


def detection(x=.5, y=.5, size=.08):
    return {"xyxy": [x-size/2,y-size/2,x+size/2,y+size/2], "confidence": .9, "label": "lug_nut"}


def jpeg_frame(seed=1):
    frame = np.random.default_rng(seed).integers(0, 255, (160, 240, 3), dtype=np.uint8)
    return frame, base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()


def test_alignment_requires_aim_fresh_stable_frames_and_reports_image_signs():
    a = Alignment()
    d = detection()
    a.select([d], [.5,.5])
    assert a.update([d], 640, 480, now=1)["state"] == "set_aim"
    a = Alignment([.5,.5])
    a.select([d], [.5,.5])
    for i in range(4):
        assert not a.update([d], 640, 480, now=1+i*.1)["image_aligned"]
    r = a.update([d], 640, 480, now=1.4)
    assert r["image_aligned"] and not r["robot_motion_enabled"]
    r = a.update([detection(.52,.51)], 640, 480, now=1.5)
    assert not r["image_aligned"]
    assert r["error_px"] == [12.8, 4.8]


@pytest.mark.parametrize("scenario,state", [("lost","target_lost"),("ambiguous","ambiguous"),("stale","stale"),("resize","camera_changed")])
def test_target_invalidations_require_explicit_reselection(scenario,state):
    a=Alignment([.5,.5]);d=detection();a.select([d],[.5,.5]);a.update([d],640,480,now=1)
    found=[] if scenario=='lost' else [d,detection(.51,.5)] if scenario=='ambiguous' else [d]
    r=a.update(found,800 if scenario=='resize' else 640,480,now=2 if scenario=='stale' else 1.1)
    assert r['state']==state and r['error_px'] is None and not r['image_aligned']
    assert a.update([d],640,480,now=2.1)['target'] is None


def test_nearby_nut_does_not_replace_missing_target():
    a=Alignment([.5,.5]);a.select([detection()],[.5,.5])
    r=a.update([detection(.65,.5)],640,480,now=1)
    assert r['state']=='target_lost'
    with pytest.raises(ValueError):
        a.select([detection(),detection()],[.5,.5])


@pytest.mark.parametrize('aim', [[float('nan'),.5], [-1,.5], [.5], [.2,.2,.2]])
def test_invalid_aim_rejected(aim):
    with pytest.raises(ValueError): Alignment(aim)


def test_dataset_saves_exact_yolo_labels_and_empty_reviewed_frames(tmp_path):
    ds=Dataset(tmp_path);frame,_=jpeg_frame()
    result=ds.save(frame,[[.2,.3,.4,.5]],'session1','train',True)
    fields=(tmp_path/'labels/train'/f"{result['saved']}.txt").read_text().split()
    np.testing.assert_allclose(list(map(float,fields)),[0,.3,.4,.2,.2])
    second,_=jpeg_frame(2)
    empty=ds.save(second,[],'session1','train',True)
    assert (tmp_path/'labels/train'/f"{empty['saved']}.txt").read_text()==''
    assert ds.status()=={'train':2,'val':0,'test':0}
    assert 'lug_nut' in (tmp_path/'dataset.yaml').read_text()


def test_dataset_rejects_leakage_unreviewed_bad_boxes_and_duplicates(tmp_path):
    ds=Dataset(tmp_path);frame,_=jpeg_frame()
    with pytest.raises(ValueError):ds.save(frame,[],'session1','train',False)
    with pytest.raises(ValueError):ds.save(frame,[[.5,.5,.2,.2]],'session1','train',True)
    with pytest.raises(ValueError):ds.save(frame,[],'../unsafe','train',True)
    ds.save(frame,[detection()['xyxy']],'session1','train',True)
    with pytest.raises(ValueError,match='already saved'):ds.save(frame,[],'session2','val',True)
    with pytest.raises(ValueError,match='another split'):ds.save(jpeg_frame(2)[0],[],'session1','val',True)


def test_training_preflight_rejects_missing_data_and_checks_full_dataset(tmp_path):
    with pytest.raises(ValueError,match='10 training'):check_dataset(tmp_path)
    ds=Dataset(tmp_path)
    for i in range(15):
        split='train' if i<10 else 'val'
        ds.save(jpeg_frame(i)[0],[detection()['xyxy']],split,split,True)
    assert check_dataset(tmp_path)['train']==10
    next((tmp_path/'labels/val').glob('*.txt')).write_text('0 NaN .5 .1 .1')
    with pytest.raises(ValueError):check_dataset(tmp_path)


def test_missing_or_generic_weights_never_produce_lug_detections(tmp_path,monkeypatch):
    detector=LugnutDetector(tmp_path/'missing.pt')
    with pytest.raises(ValueError,match='No trained'):detector.predict(jpeg_frame()[0])
    weights=tmp_path/'generic.pt';weights.write_bytes(b'test')
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda *a,**k:SimpleNamespace(names={0:'person',1:'car'})))
    detector=LugnutDetector(weights)
    with pytest.raises(ValueError,match='lug_nut'):detector.predict(jpeg_frame()[0])
    assert not detector.status()['loaded']


def test_vision_api_capture_origin_and_input_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(vision_routes,'dataset',Dataset(tmp_path))
    frame,b64=jpeg_frame();payload={'frame':b64,'boxes':[detection()['xyxy']],'session':'camera1','split':'train','reviewed':True}
    with TestClient(app) as client:
        assert client.get('/align').status_code==200
        assert client.get('/alignment.js').status_code==200
        assert not client.get('/vision/status').json()['detector']['robot_motion_enabled']
        assert client.post('/vision/examples',json=payload).status_code==403
        headers={'origin':'http://testserver'}
        assert client.post('/vision/examples',json=payload,headers=headers).status_code==200
        assert client.post('/vision/examples',json=[],headers=headers).status_code==400
        assert client.post('/vision/examples',content=b'x'*3_000_001,headers=headers).status_code==413


def test_detection_socket_alignment_and_loss(monkeypatch):
    found=[detection()]
    monkeypatch.setattr(vision_routes.detector,'predict',lambda *args:found)
    _,b64=jpeg_frame()
    with TestClient(app) as client:
        with client.websocket_connect('/vision/detect',headers={'origin':'http://testserver'}) as socket:
            socket.send_json({'type':'configure','aim':[.5,.5]})
            assert socket.receive_json()['type']=='configured'
            socket.send_json({'type':'frame','frame_id':1,'frame':b64})
            assert len(socket.receive_json()['detections'])==1
            socket.send_json({'type':'select','frame_id':1,'point':[.5,.5]})
            assert socket.receive_json()['type']=='selected'
            for i in range(2,7):
                socket.send_json({'type':'frame','frame_id':i,'frame':b64})
                r=socket.receive_json()
            assert r['alignment']['image_aligned']
            found.clear()
            socket.send_json({'type':'frame','frame_id':7,'frame':b64})
            assert socket.receive_json()['alignment']['state']=='target_lost'
            socket.send_json({'type':'frame','frame_id':7,'frame':b64})
            assert socket.receive_json()['type']=='error'
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/vision/detect',headers={'origin':'https://example.com'}):pass


def test_socket_reports_missing_model_and_rejects_invalid_config(monkeypatch,tmp_path):
    monkeypatch.setattr(vision_routes,'detector',LugnutDetector(tmp_path/'none.pt'))
    with TestClient(app) as client:
        with client.websocket_connect('/vision/detect',headers={'origin':'http://testserver'}) as socket:
            socket.send_json({'type':'configure','aim':[float('inf'),.5]})
            assert socket.receive_json()['type']=='error'
            socket.send_json({'type':'frame','frame_id':1,'frame':jpeg_frame()[1]})
            r=socket.receive_json()
            assert r['type']=='error' and 'No trained' in r['message']
