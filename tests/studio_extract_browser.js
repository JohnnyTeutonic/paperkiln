"use strict";
// Optional offline browser integration; every HTTP response is intercepted.
const {chromium}=require("playwright");
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
(async()=>{
 const html=fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8");
 const out=process.argv[2];if(out)fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({channel:"chrome",headless:true,args:["--disable-gpu","--disable-background-networking","--disable-extensions"]});
 const page=await browser.newPage({viewport:{width:1200,height:1000}});
 const errors=[];page.on("pageerror",e=>errors.push(e.message));
 let state="idle",job=0,mode="success",polls=0;const calls=[];
 const result={arxiv_id:"1234.56789",title:'<img src=x onerror="bad()">',fields:{d_model:{value:16,verdict:"used",evidence:"dimension 16"},norm:{value:"rmsnorm",verdict:"unverified",evidence:"invented quote"}},unresolved:["n_layers"],mechanism_scan:[{blocking:true,mechanism:"novel"}],llm:{verification:{summary:{unverified:1}}}};
 await page.route("**/*",async route=>{
  const request=route.request(),url=new URL(request.url()),name=url.pathname;
  calls.push(name);
  if(name==="/")return route.fulfill({contentType:"text/html",body:html});
  if(name==="/extract"){
   assert.equal(request.method(),"POST");assert.equal(request.postDataJSON().budget_usd,1);
   state="extracting";polls=0;job++;return route.fulfill({status:202,body:"extracting"});
  }
  if(name==="/extractcancel"){state="cancelled";return route.fulfill({body:state});}
  if(name==="/extractstatus"){
   if(state==="extracting"&&mode!=="running"&&++polls>1)state=mode==="success"?"done":"failed";
   return route.fulfill({contentType:"application/json",body:JSON.stringify({state,job:String(job),configured:true,error:state==="failed"?"mock provider refused":undefined})});
  }
  if(name==="/extractresult")return route.fulfill({contentType:"application/json",body:JSON.stringify(result)});
  return route.fulfill({status:404,body:""});
 });
 try{
  await page.goto("http://studio.test/");await page.locator("#paper").waitFor({state:"visible"});
  const before=await page.locator("#b-d").inputValue();
  await page.locator("#p-id").fill("1234.56789");await page.locator("#p-budget").fill("1");
  await page.locator("#p-go").click();await page.locator("#p-result details").waitFor();
  assert.equal(await page.locator("#p-result img").count(),0);assert.equal(await page.locator("#b-d").inputValue(),before);
  assert.ok((await page.locator("#p-result").textContent()).includes("unverified"));
  await page.locator("#p-result summary").click();assert.ok((await page.locator("#p-result pre").textContent()).includes('"blocking": true'));
  if(out)await page.locator("#paper").screenshot({path:path.join(out,"llm_review.png")});
  await page.setViewportSize({width:420,height:1000});
  assert.equal(await page.locator("#paper").evaluate(e=>e.scrollWidth>e.clientWidth+2),false,"extractor panel overflow");
  if(out)await page.locator("#paper").screenshot({path:path.join(out,"llm_review_narrow.png")});
  mode="failure";await page.locator("#p-go").click();await page.locator("#p-status").filter({hasText:"mock provider refused"}).waitFor();
  assert.equal(await page.locator("#p-result details").count(),0);
  mode="running";await page.locator("#p-go").click();await page.locator("#p-cancel").waitFor({state:"visible"});await page.locator("#p-cancel").click();
  await page.locator("#p-status").filter({hasText:"cancelled"}).waitFor();
  assert.equal(calls.filter(x=>x==="/fetch").length,0);assert.deepEqual(errors,[]);
  console.log(JSON.stringify({status:"PASS",cases:["default LLM mode","actual report review","HTML injection stays text","builder unchanged","failure without fallback","cancel"],billable_calls:0,calls}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
