"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const html=fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8");
const ctx={};vm.runInNewContext(html.split("// BEGIN STUDIO_VRAM_CORE")[1].split("// END STUDIO_VRAM_CORE")[0]+"\nthis.api={estimateVram,planVram}",ctx);
const {estimateVram,planVram}=ctx.api;
function spec(preset="llama-tiny"){return {arch:{preset,custom:{d:16,layers:2,heads:2}},data:{T:8,vocab_cap:64},train:{optimizer:"adamw",batch:4,accum:2}};}
test("parameter inventory includes tied heads/norms/biases and custom FFN",()=>{
 assert.equal(estimateVram(spec(),0).params,7760);
 assert.equal(estimateVram(spec("gpt2-nano"),0).params,7744);
 const s=spec();s.arch.custom.d_ff=64;assert.equal(estimateVram(s,0).params,9296);
});
test("formula breakdown is explicit and accumulation does not retain another graph",()=>{
 const s=spec(),e=estimateVram(s,.1);assert.equal(e.parts.attention,16*2*2*(4*8)**2);assert.equal(e.parts.logits,32*4*8*64);
 assert.equal(e.parts.parameters,32*e.params);s.train.accum=8;assert.equal(estimateVram(s,.1).total,e.total);
});
test("memory increases monotonically with microbatch/context/vocab/depth",()=>{
 for(const edit of [s=>s.train.batch++,s=>s.data.T++,s=>s.data.vocab_cap++,s=>s.arch.custom.layers++]){const s=spec(),base=estimateVram(s,0).total;edit(s);assert.ok(estimateVram(s,0).total>base);}
});
test("exact boundary accepted; divisor proposal preserves effective batch and input",()=>{
 const s=spec(),original=JSON.stringify(s),half=spec();half.train.batch=2;
 const limit=estimateVram(half,0).total/2**30,p=planVram(s,limit,0);
 assert.equal(p.batch,2);assert.equal(p.accum,4);assert.equal(JSON.stringify(s),original);assert.equal(p.estimate.T,8);
 assert.equal(planVram(s,estimateVram(s,0).total/2**30,0).batch,4);
 assert.equal(planVram(s,1e-10,0).estimate,null);
});
test("unsupported execution paths are refused",()=>{
 for(const edit of [s=>s.arch.preset="kimi-tiny",s=>s.train.optimizer="muon",s=>s.train.precision="bf16",s=>s.train.checkpoint_activations=true,s=>s.arch.custom.norm="rmsnorm",s=>s.arch.custom.bitlinear=true,s=>s.arch.recursion={passes:2},s=>s.train.attention_kernel="flash",s=>s.arch.custom.heads=3]){const s=spec();edit(s);assert.throws(()=>estimateVram(s,0));}
 const s=spec("gpt2-nano");s.arch.custom.d_ff=64;assert.throws(()=>estimateVram(s,0));
});
test("invalid, unsafe and unbounded inputs fail closed",()=>{
 for(const x of [NaN,Infinity,-1,0]) assert.throws(()=>planVram(spec(),x,0));
 for(const x of [NaN,Infinity,-1]) assert.throws(()=>estimateVram(spec(),x));
 for(const x of [0,-1,1.5,Number.MAX_SAFE_INTEGER]) {const s=spec();s.data.T=x;assert.throws(()=>estimateVram(s,0));}
 const s=spec();s.train.batch=65537;assert.throws(()=>planVram(s,8,0));
});
