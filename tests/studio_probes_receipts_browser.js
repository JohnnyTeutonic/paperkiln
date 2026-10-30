"use strict";
const {chromium}=require("playwright"),fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict"),crypto=require("node:crypto");
(async()=>{
 const root=path.join(__dirname,".."),html=fs.readFileSync(path.join(root,"studio/index.html"),"utf8");
 const receipts=["wordlevel.json","native_bpe.json","oversized_error.json"].map(name=>{const bytes=fs.readFileSync(path.join(root,"docs/receipts/studio_probes_20261031",name));return {name,bytes,report:JSON.parse(bytes),sha256:crypto.createHash("sha256").update(bytes).digest("hex")};});
 const browser=await chromium.launch({channel:"chrome",headless:true,args:["--disable-gpu","--disable-background-networking"]}),page=await browser.newPage({viewport:{width:1200,height:1000}}),errors=[],calls=[];let current;
 page.on("pageerror",e=>errors.push(e.message));await page.route("**/*",route=>{const url=new URL(route.request().url());calls.push(url.pathname);if(url.pathname==="/")return route.fulfill({contentType:"text/html",body:html});if(url.pathname==="/evaluation-probes"&&current)return route.fulfill({contentType:"application/json",body:current.bytes});return route.fulfill({status:404,body:""});});
 try{
  await page.goto("http://studio.test/");await page.locator("#probe-panel summary").click();const initial=await page.locator("#b-json").textContent(),results=[];
  for(current of receipts){
   await page.locator("#probe-refresh").click();await page.waitForFunction(expected=>document.getElementById("probe-results").textContent===JSON.stringify(expected,null,2),current.report);
   assert.deepEqual(JSON.parse(await page.locator("#probe-results").textContent()),current.report);assert.equal(await page.locator("#b-json").textContent(),initial);
   const status=await page.locator("#probe-result-status").textContent();assert.ok(status.includes(current.report.status==="ERROR"?"Operational probe error":current.report.status));
   results.push({file:current.name,sha256:current.sha256,recorded_status:current.report.status,viewer_status:status});
  }
  assert.deepEqual(errors,[]);assert.equal(calls.some(x=>["/train","/sample","/extract"].includes(x)),false);
  console.log(JSON.stringify({status:"PASS",receipts:results,source_sha256:crypto.createHash("sha256").update(html).digest("hex"),provider_calls:0,gpu_calls:0,additional_model_calls:0}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
