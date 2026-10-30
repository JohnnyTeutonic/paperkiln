"use strict";
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8");
const core = html.split("// BEGIN STUDIO_RESEARCH_CORE")[1].split("// END STUDIO_RESEARCH_CORE")[0];
const context = {module:{exports:{}}, JSON}; vm.runInNewContext(core,context);
const r = context.module.exports;
const spec = {name:"original",out_dir:"/tmp/original",arch:{custom:{d:16},unknown:{nested:[true,null,"keep"]}},train:{lr:0.001,steps:6,seed:0},opaque:{z:0}};
const lines = xs => xs.map(x => JSON.stringify(x)).join("\n");
const step = {event:"step",step:1,loss:2};

test("clone changes only chosen field and administrative identity",()=>{
  const before=JSON.stringify(spec); const copy=r.clone(spec,["train","lr"],0.002,"clone-test");
  assert.equal(JSON.stringify(spec),before);
  assert.equal(copy.train.lr,0.002);assert.equal(copy.name,"original-clone-test");assert.equal(copy.out_dir,"/tmp/original-clone-test");
  copy.train.lr=spec.train.lr;copy.name=spec.name;copy.out_dir=spec.out_dir;
  assert.equal(JSON.stringify(copy),before);
});
test("arrays, false, null and unknown fields survive",()=>{
  const copy=r.clone(spec,["arch","unknown","nested","0"],false,"clone");
  assert.equal(copy.arch.unknown.nested[0],false);assert.equal(copy.arch.unknown.nested[1],null);assert.equal(copy.opaque.z,0);
});
for(const p of [["train","missing"],["train","length"],["arch","unknown","nested","length"],["__proto__","polluted"],["name"],["out_dir"],["constructor"]]){
  test("unlisted or dangerous path refuses: "+JSON.stringify(p),()=>assert.throws(()=>r.clone(spec,p,1,"clone")));
}
test("type changes, same value, fractional integers and unsafe numbers refuse",()=>{
  for(const [p,v] of [[["train","lr"],"0.002"],[["train","lr"],.001],[["train","steps"],2.5],[["train","steps"],Infinity],[["train","steps"],2**53],[["train","lr"],-0]]) assert.throws(()=>r.clone(spec,p,v,"clone"));
  assert.throws(()=>r.fields({train:{seed:2**53}}));assert.throws(()=>r.fields({train:{lr:-0}}));
});
test("strict logs reject malformed records and nonfinite losses",()=>{
  for(const text of ['{bad}',lines([{step:1}]),lines([{event:"step",step:1,loss:null}]),lines([{event:"step",step:1.5,loss:2}]),lines([{event:"start",name:"empty"}])]) assert.throws(()=>r.events(text,"bad.jsonl"));
});
test("early stop stays early stop and a resumed run becomes in progress",()=>{
  assert.equal(r.events(lines([step,{event:"early_stop",step:1},{event:"done",early_stopped:true}]),"A").status,"early stopped");
  assert.equal(r.events(lines([step,{event:"done"},{event:"start",name:"resumed"},{...step,step:2}]),"A").status,"in progress");
});
test("restarts are retained and flagged rather than sorted away",()=>{
  const got=r.events(lines([step,{...step,step:3},{...step,step:2}]),"A");assert.equal(got.resets,1);assert.equal(got.steps[2].step,2);assert.equal(got.steps[2].reset,true);
});
test("large logs do not overflow argument limits",()=>{
  const text=Array.from({length:150000},(_,i)=>JSON.stringify({event:"eval",step:i,val_loss:1+i/150000})).join("\n");
  const got=r.events(text,"large.jsonl");assert.equal(got.evals.length,150000);assert.equal(got.best,1);
});
test("untrusted names remain data",()=>assert.equal(r.events(lines([{event:"start",name:'<img src=x onerror="bad()">'},step]),"A").name,'<img src=x onerror="bad()">'));