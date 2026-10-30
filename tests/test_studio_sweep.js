"use strict";
const {test}=require("node:test");const assert=require("node:assert/strict");
const fs=require("node:fs");const path=require("node:path");const vm=require("node:vm");
const html=fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8");
const context={module:{exports:{}}};vm.runInNewContext(html.split("// BEGIN STUDIO_SWEEP_CORE")[1].split("// END STUDIO_SWEEP_CORE")[0],context);
const s=context.module.exports;
const cell=(id,x,y,more={})=>({cell:id,factors:{x,y},n_seeds:3,best_val_mean:2+id/10,best_val_std:.1,best_val_min:1,...more});
const encode=xs=>xs.map(x=>JSON.stringify(x)).join("\n");
const parse=xs=>s.parse(encode(xs));
test("reads the mtsweep aggregate schema without changing saved statistics",()=>{
 const data=parse([cell(0,.001,"exact"),cell(1,.003,"exact"),cell(2,.001,"kimi")]);
 const v=s.slice(data,"x","y",{});assert.equal(v.selected,3);assert.equal(v.min,2);assert.equal(v.max,2.2);assert.equal(v.xs[0],.001);
 assert.equal(v.buckets.get(JSON.stringify(["0.001",'"exact"']))[0].n_seeds,3);
});
test("unobserved, zero-seed and repeated-coordinate cells remain distinct",()=>{
 const data=parse([cell(0,0,false),cell(1,0,false),cell(2,1,true,{n_seeds:0,best_val_mean:null,best_val_std:null})]);
 const v=s.slice(data,"x","y",{});assert.equal(v.buckets.get(JSON.stringify(["0","false"])).length,2);
 assert.equal(v.buckets.get(JSON.stringify(["1","true"]))[0].n_seeds,0);assert.equal(v.buckets.has(JSON.stringify(["0","true"])),false);assert.equal(v.min,null);
});
test("typed factor levels do not collapse false, zero or string zero",()=>{
 const data=parse([cell(0,false,0),cell(1,0,0),cell(2,"0",0)]);assert.equal(s.levels(data,"x").length,3);
});
test("all nuisance factors must be fixed explicitly; no implicit marginal pooling",()=>{
 const a=cell(0,1,2);a.factors.z="a";const b=cell(1,1,2);b.factors.z="b";const data=parse([a,b]);
 assert.throws(()=>s.slice(data,"x","y",{}),/every other/);assert.throws(()=>s.slice(data,"x","y",{z:'"absent"'}),/every other/);
 assert.equal(s.slice(data,"x","y",{z:'"a"'}).selected,1);assert.equal(s.slice(data,"x","y",{z:'"b"'}).min,2.1);
});
test("linked factor objects use stable exact JSON identity",()=>{
 assert.equal(s.key({b:false,a:1}),s.key({a:1,b:false}));assert.notEqual(s.key({x:1}),s.key({x:"1"}));
 const data=parse([cell(0,{b:2,a:1},1),cell(1,{a:1,b:2},2)]);assert.equal(s.levels(data,"x").length,1);
});
test("malformed JSON, empty files and partial schemas refuse",()=>{
 assert.throws(()=>s.parse("bad"),/Invalid JSON/);assert.throws(()=>s.parse("\n"),/No cell/);assert.throws(()=>parse([{}]),/Invalid cell/);
 assert.throws(()=>parse([{...cell(0,1,2),factors:{x:1}}]),/two factors/);
 assert.throws(()=>parse([cell(0,1,2),{...cell(1,1,2),factors:{a:1,b:2}}]),/same factors/);
});
test("duplicate IDs and invalid numeric receipt fields refuse",()=>{
 assert.throws(()=>parse([cell(0,1,2),cell(0,2,3)]),/Duplicate/);
 for(const bad of [{n_seeds:-1},{n_seeds:1.5},{cell:1.5},{best_val_mean:null},{best_val_std:-.1},{n_seeds:0,best_val_mean:2}]) assert.throws(()=>parse([cell(0,1,2,bad)]));
 assert.throws(()=>s.parse('{"cell":0,"factors":{"x":1,"y":2},"n_seeds":1,"best_val_mean":1e999}'),/finite/);
});
test("axis selection refuses same/unknown axes and oversized grids",()=>{
 const data=parse([cell(0,1,2)]);assert.throws(()=>s.slice(data,"x","x",{}),/distinct/);assert.throws(()=>s.slice(data,"x","missing",{}),/distinct/);
 const rows=Array.from({length:31},(_,i)=>cell(i,i,i));assert.throws(()=>s.slice(parse(rows),"x","y",{}),/900/);
});
test("unsafe factor integers and prototype names refuse, HTML names stay data",()=>{
 assert.throws(()=>parse([cell(0,9007199254740992,1)]),/exactly/);
 assert.throws(()=>s.parse('{"cell":0,"factors":{"__proto__":1,"x":2},"n_seeds":0}'),/factor name/);
 const data=parse([cell(0,'<img src=x onerror="bad()">',1)]);assert.equal(data.cells[0].factors.x,'<img src=x onerror="bad()">');
});