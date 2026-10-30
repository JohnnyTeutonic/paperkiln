"use strict";
const {chromium}=require("playwright"),fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
(async()=>{
 const html=fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8");
 const browser=await chromium.launch({channel:"chrome",headless:true,args:["--disable-gpu","--disable-background-networking"]});
 const page=await browser.newPage({viewport:{width:1200,height:1000}}),errors=[];page.on("pageerror",e=>errors.push(e.message));
 await page.route("**/*",route=>new URL(route.request().url()).pathname==="/"?route.fulfill({contentType:"text/html",body:html}):route.fulfill({status:404,body:""}));
 try{
  await page.goto("http://studio.test/");await page.locator("#vram-panel summary").click();
  await page.locator("#vram-reserve").fill("0");
  const initial=await page.locator("#b-json").textContent();
  const limit=await page.evaluate(()=>{const s=JSON.parse(buildSpec());s.train.batch=2;return estimateVram(s,0).total/2**30;});
  await page.locator("#vram-gib").fill(String(limit));await page.locator("#vram-plan").click();
  assert.equal(await page.locator("#b-batch").inputValue(),"4");assert.equal(await page.locator("#b-json").textContent(),initial);
  assert.equal(await page.locator("#vram-apply").isEnabled(),true);await page.locator("#vram-apply").click();
  const applied=JSON.parse(await page.locator("#b-json").textContent()),original=JSON.parse(initial);
  assert.equal(applied.train.batch,2);assert.equal(applied.train.accum,2);original.train.batch=2;original.train.accum=2;assert.deepEqual(applied,original);
  await page.locator("#b-batch").fill("4");await page.locator("#vram-plan").click();
  await page.locator("#b-T").fill("128");assert.equal(await page.locator("#vram-apply").isDisabled(),true);
  await page.locator("#b-ckpt").check();await page.locator("#vram-plan").click();assert.ok((await page.locator("#vram-result").textContent()).includes("Unsupported"));
  await page.setViewportSize({width:420,height:1000});assert.equal(await page.locator("#vram-panel").evaluate(e=>e.scrollWidth>e.clientWidth+2),false);
  assert.deepEqual(errors,[]);console.log(JSON.stringify({status:"PASS",cases:["preview leaves spec unchanged","explicit apply changes batch/accum only","effective batch preserved","stale proposal disabled","checkpoint refused","420px panel fits"],provider_calls:0,gpu_calls:0}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
