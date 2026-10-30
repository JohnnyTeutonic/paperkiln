"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm"),path=require("node:path"),{TextEncoder}=require("node:util");
const html=fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8"),ctx={TextEncoder};
vm.runInNewContext(html.split("// BEGIN STUDIO_PROBES_CORE")[1].split("// END STUDIO_PROBES_CORE")[0]+"\nthis.api={validateProbeConfig,probeContract,validateProbeReport,renderProbeReport}",ctx);
const {validateProbeConfig,probeContract,validateProbeReport,renderProbeReport}=ctx.api;
function config(){return {max_new_tokens:16,cases:[{id:"needle1",kind:"needle",prompt:"secret is amber",contract:"contains",expected:"amber"}]};}
function report(){const configuration=config();return {schema_version:1,status:"COMPLETE",configuration,generation:{mode:"greedy"},cases:[{...configuration.cases[0],status:"PASS",generated_text:"amber"}],summary:{passed:1,failed:0,errors:0,total:1}};}
test("configuration copied and exact literal rules preserve case/whitespace",()=>{
 const c=config(),v=validateProbeConfig(c);v.cases[0].prompt="changed";assert.equal(c.cases[0].prompt,"secret is amber");
 assert.equal(probeContract(" OK","exact","OK"),false);assert.equal(probeContract("OK!","prefix","OK"),true);
 assert.equal(probeContract("amber","contains","Amber"),false);assert.equal(probeContract("amber","not_contains","red"),true);
});
test("bounded configuration, unknown keys, duplicates and UTF8 byte limits refused",()=>{
 for(const edit of [c=>c.max_new_tokens=129,c=>c.max_new_tokens=0,c=>c.max_new_tokens=1.1,c=>c.shell="bad",c=>c.cases=[],c=>c.cases[0].id="../bad",c=>c.cases[0].kind="semantic",c=>c.cases[0].contract="regex",c=>c.cases[0].expected="",c=>c.cases[0].expected="missing",c=>c.cases[0].contract="not_contains",c=>c.cases[0].prompt="\ud800",c=>c.cases[0].prompt="😀".repeat(2049),c=>c.cases[0].expected="😀".repeat(65),c=>c.cases[0].extra=true,c=>c.cases.push({...c.cases[0]})]){const c=config();edit(c);assert.throws(()=>validateProbeConfig(c));}
 const c=config();c.max_new_tokens=128;c.cases=Array.from({length:9},(_,i)=>({...c.cases[0],id:"c"+i}));assert.throws(()=>validateProbeConfig(c));
 c.cases.pop();assert.equal(validateProbeConfig(c).cases.length,8);
});
test("results bind config/outcomes/summary and distinguish operational errors",()=>{
 assert.equal(validateProbeReport(report()).summary.passed,1);
 for(const edit of [r=>r.cases[0].id="other",r=>r.cases[0].generated_text="not found",r=>r.summary.passed=0,r=>r.cases=[],r=>r.generation.mode="sample",r=>r.status="COMPLETE_WITH_ERRORS"]){const r=report();edit(r);assert.throws(()=>validateProbeReport(r));}
 const r=report();r.cases[0].status="ERROR";r.cases[0].error="prompt too long";r.status="COMPLETE_WITH_ERRORS";r.summary={passed:0,failed:0,errors:1,total:1};assert.equal(validateProbeReport(r).status,"COMPLETE_WITH_ERRORS");
});
test("untrusted report text rendered literally with no HTML insertion",()=>{
 const elements={};ctx.$=id=>elements[id]??={textContent:""};const r=report();r.configuration.cases[0].prompt=r.cases[0].prompt=r.configuration.cases[0].expected=r.cases[0].expected=r.cases[0].generated_text='<img src=x onerror="bad()">';
 renderProbeReport(r);assert.ok(elements["probe-results"].textContent.includes("<img"));assert.ok(elements["probe-result-status"].textContent.includes("not semantic"));
});
test("validated post-training stage accepted by training-only VRAM planner",()=>{
 vm.runInNewContext(html.split("// BEGIN STUDIO_VRAM_CORE")[1].split("// END STUDIO_VRAM_CORE")[0]+"\nthis.estimateVram=estimateVram",ctx);
 const s={arch:{preset:"llama-tiny"},data:{T:8,vocab_cap:64},train:{batch:2,accum:1}},base=ctx.estimateVram(s,0).total;
 s.evaluation_probes=config();assert.equal(ctx.estimateVram(s,0).total,base);
 s.evaluation_probes.max_new_tokens=999;assert.throws(()=>ctx.estimateVram(s,0));
});
