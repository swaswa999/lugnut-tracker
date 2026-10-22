const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname,'../alignment.js'),'utf8');
const settle = () => new Promise(resolve=>setImmediate(resolve));

function setup(weights=true) {
  const nodes=new Map(), state={now:100,streams:[],sockets:[],requests:[],listeners:{}};
  function element() {
    return {value:'',textContent:'',children:[],width:960,height:600,readyState:2,videoWidth:640,videoHeight:480,
      paused:true,ended:false,currentTime:1,checked:false,disabled:false,hidden:false,
      addEventListener(){},removeAttribute(){},load(){},setAttribute(){},setPointerCapture(){},
      pause(){this.paused=true;},async play(){this.paused=false;},
      replaceChildren(...items){this.children=items;},add(item){this.children.push(item);},
      getBoundingClientRect(){return {left:0,top:0,width:640,height:480};},
      getContext(){return new Proxy({},{get:()=>()=>{}});},toDataURL(){return 'data:image/jpeg;base64,dGVzdA==';}};
  }
  const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  get('confidence').value='.5';get('tolerance').value='5';get('session').value='session1';get('split').value='train';
  const media={async enumerateDevices(){return [{kind:'videoinput',deviceId:'usb',label:'USB'}];},
    addEventListener(){},async getUserMedia(){const track={stopped:0,stop(){this.stopped++;}};
      const stream={getTracks:()=>[track],getVideoTracks:()=>[track],track};state.streams.push(stream);return stream;}};
  class WS {
    static OPEN=1;
    constructor(){this.readyState=1;this.sent=[];state.sockets.push(this);}
    send(data){this.sent.push(JSON.parse(data));}
    close(){this.readyState=3;}
    receive(data){this.onmessage({data:JSON.stringify(data)});}
  }
  const document={hidden:false,getElementById:get,createElement:element,
    addEventListener(name,fn){state.listeners[name]=fn;}};
  const context=vm.createContext({console,performance:{now:()=>state.now},WebSocket:WS,
    requestAnimationFrame(fn){state.draw=fn;},navigator:{mediaDevices:media},document,
    window:{addEventListener(){}},location:{host:'localhost:8767'},URL:{revokeObjectURL(){}},
    Option:function(text,value){this.text=text;this.value=value;},
    async fetch(url,options) {
      state.requests.push({url,options});
      return {ok:true,async json() {
        return url==='/vision/status'?{detector:{weights_available:weights},dataset:{train:0,val:0,test:0}}:{saved:'example'};
      }};
    }
  });
  vm.runInContext(code,context);
  return {state,nodes,media,document,get,click:async id=>{await get(id).onclick();await settle();}};
}

test('pending camera access is cancelled and late stream is released',async()=>{
  const t=setup();await settle();let finish;
  t.media.getUserMedia=()=>new Promise(resolve=>{finish=resolve;});
  const pending=t.click('camera');await settle();await t.click('stop');
  let stopped=0;const track={stop(){stopped++;}};
  finish({getTracks:()=>[track],getVideoTracks:()=>[track]});await pending;
  assert.equal(stopped,1);assert.equal(t.get('source').srcObject,null);
});

test('drawn reviewed boxes are saved with the frozen image',async()=>{
  const t=setup();await settle();await t.click('camera');await t.click('freeze');
  t.get('view').onpointerdown({clientX:64,clientY:96,pointerId:1});
  t.get('view').onpointerup({clientX:192,clientY:192});
  await t.click('save');assert.match(t.get('message').textContent,/Review/);
  t.get('reviewed').checked=true;await t.click('save');
  const request=t.state.requests.find(r=>r.url==='/vision/examples');
  const body=JSON.parse(request.options.body);
  assert.deepEqual(body.boxes,[[.1,.2,.3,.4]]);assert.equal(body.frame,'dGVzdA==');assert.equal(body.reviewed,true);
});

test('missing model disables detection and gives a useful next step',async()=>{
  const t=setup(false);await settle();await t.click('alignTab');await t.click('detect');
  assert.equal(t.get('detect').disabled,true);assert.equal(t.state.sockets.length,0);
  assert.match(t.get('modelStatus').textContent,/No trained model/);
});

async function live() {
  const t=setup();await settle();await t.click('camera');await t.click('alignTab');await t.click('detect');
  const socket=t.state.sockets[0];socket.onopen();socket.receive({type:'configured'});t.state.draw();
  return {t,socket};
}
function result(id=1) {return {type:'result',frame_id:id,processing_ms:30,detections:[],
  alignment:{state:'image_aligned',image_aligned:true,error_px:[0,0],stable_frames:5}};}

test('stale inference result never remains as centered guidance',async()=>{
  const {t,socket}=await live();t.state.now+=800;socket.receive(result());
  assert.equal(t.get('error').textContent,'—');assert.match(t.get('state').textContent,/too old/);
  assert.equal(socket.sent.at(-1).type,'clear');
});

test('repeated video frame is not resubmitted and guidance expires',async()=>{
  const {t,socket}=await live();t.state.now+=30;socket.receive(result());
  assert.equal(t.get('state').textContent,'Image centered');
  t.state.now+=300;t.state.draw();assert.equal(socket.sent.filter(x=>x.type==='frame').length,1);
  t.state.now+=800;t.state.draw();assert.equal(t.get('error').textContent,'—');
});

test('configuration change drops in-flight results until acknowledgement',async()=>{
  const {t,socket}=await live();t.get('tolerance').value='3';t.get('tolerance').onchange();
  socket.receive(result());assert.equal(t.get('error').textContent,'—');
  t.get('source').currentTime=2;t.state.now+=120;t.state.draw();
  assert.equal(socket.sent.filter(x=>x.type==='frame').length,1);
  socket.receive({type:'configured'});t.state.draw();
  assert.equal(socket.sent.filter(x=>x.type==='frame').length,2);
});

test('hidden tab closes inference and clears guidance',async()=>{
  const {t,socket}=await live();socket.receive(result());
  t.document.hidden=true;t.state.listeners.visibilitychange();
  assert.equal(socket.readyState,3);assert.equal(t.get('error').textContent,'—');
});
