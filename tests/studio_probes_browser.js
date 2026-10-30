"use strict";
const {chromium}=require("playwright"),fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
(async()=>{
 const html=fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8"),browser=await chromium.launch({channel:"chrome",headless:true,args:["--disable-gpu","--disable-background-networking"]});
 const page=await browser.newPage({viewport:{width:1200,height:1000}}),errors=[],calls=[];page.on("pageerror",e=>errors.push(e.message));
 let reply=null;
 await page.route("**/*",route=>{const url=new URL(route.request().url());calls.push(url.pathname);if(url.pathname==="/")return route.fulfill({contentType:"text/html",body:html});if(url.pathname==="/evaluation-probes"&&reply)return route.fulfill({contentType:"application/json",body:JSON.stringify(reply)});return route.fulfill({status:404,body:""});});
 try{
  await page.goto("http://studio.test/");await page.locator("#probe-panel summary").click();
  const initial=JSON.parse(await page.locator("#b-json").textContent());assert.equal(initial.evaluation_probes,undefined);
  await page.locator("#probe-enabled").check();assert.equal(JSON.parse(await page.locator("#b-json").textContent()).evaluation_probes,undefined);
  await page.locator("#probe-apply").click();const applied=JSON.parse(await page.locator("#b-json").textContent());assert.equal(applied.evaluation_probes.cases.length,2);
  delete applied.evaluation_probes;assert.deepEqual(applied,initial);
  const config=JSON.parse(await page.locator("#probe-config").inputValue());await page.locator("#probe-config").fill('{"unknown":true}');await page.locator("#probe-apply").click();
  assert.equal(JSON.parse(await page.locator("#b-json").textContent()).evaluation_probes.cases.length,2);assert.ok((await page.locator("#probe-status").textContent()).includes("preserved"));
  reply={schema_version:1,status:"COMPLETE",configuration:config,generation:{mode:"greedy",context_policy:"reject_long_prompt_then_sliding_window"},cases:config.cases.map((c,i)=>({...c,status:i?"FAIL":"PASS",generated_text:i?'<img src=x onerror="bad()">':"amber",stop_reason:"token_limit",context_evicted_tokens:0,generated_ids:[1],prompt_ids:[2],generated_tokens:1,prompt_tokens:1})),summary:{passed:1,failed:1,errors:0,total:2},checkpoint:{path:"checkpoint.safetensors",bytes:10,sha256:"a".repeat(64)}};
  await page.locator("#probe-refresh").click();await page.locator("#probe-result-status").filter({hasText:"1 literal passes"}).waitFor();assert.equal(await page.locator("#probe-results img").count(),0);assert.ok((await page.locator("#probe-results").textContent()).includes("<img"));
  reply.summary.passed=2;await page.locator("#probe-refresh").click();await page.locator("#probe-result-status").filter({hasText:"differs"}).waitFor();assert.equal(await page.locator("#probe-results").textContent(),"");
  await page.locator("#probe-enabled").uncheck();await page.locator("#probe-apply").click();assert.equal(JSON.parse(await page.locator("#b-json").textContent()).evaluation_probes,undefined);
  await page.setViewportSize({width:420,height:1000});assert.equal(await page.locator("#probe-panel").evaluate(e=>e.scrollWidth>e.clientWidth+2),false);
  assert.deepEqual(errors,[]);assert.equal(calls.some(x=>["/train","/sample","/extract"].includes(x)),false);
  console.log(JSON.stringify({status:"PASS",cases:["absent by default","explicit application only","invalid draft preserves applied stage","literal PASS/FAIL raw text review","HTML injection stays text","tampered summary rejected/no stale output","disable removes stage","420px fits"],provider_calls:0,gpu_calls:0,model_calls:0}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
