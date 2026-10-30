"use strict";
// Optional browser smoke: NODE_PATH must resolve Playwright; uses installed Chrome, no GPU/network/model calls.
const {chromium}=require("playwright");
const fs=require("node:fs");const path=require("node:path");const os=require("node:os");const crypto=require("node:crypto");const assert=require("node:assert/strict");const {pathToFileURL}=require("node:url");
(async()=>{
  const out=path.resolve(process.argv[2] || path.join(os.tmpdir(),"paperkiln-studio-research"));fs.mkdirSync(out,{recursive:true});
  const original=process.env.STUDIO_SPEC ? JSON.parse(fs.readFileSync(process.env.STUDIO_SPEC,"utf8")) : {name:"fixture",out_dir:"fixture-out",arch:{family:"gpt2",custom:{d:16,layers:2,heads:2}},data:{vocab:"/path/to/vocab.gguf",corpus:"/path/to/corpus.txt",vocab_cap:4096,T:32},train:{steps:6,lr:.001,batch:1,seed:841},export:{formats:["safetensors","gguf"]}};
  if(process.env.STUDIO_OUTPUT_BASE) original.out_dir=process.env.STUDIO_OUTPUT_BASE;
  original.research_annotation={source:"browser smoke",keep:[false,null,0]};
  fs.writeFileSync(path.join(out,"original_spec.json"),JSON.stringify(original,null,2)+"\n");
  const browser=await chromium.launch({...(process.env.STUDIO_CHROME ? {executablePath:process.env.STUDIO_CHROME} : {channel:"chrome"}),headless:true,args:["--disable-gpu","--disable-background-networking","--disable-extensions"]});
  const context=await browser.newContext({acceptDownloads:true,viewport:{width:1160,height:1000}});const page=await context.newPage();const errors=[];page.on("pageerror",e=>errors.push(e.message));
  try{
    await page.goto(pathToFileURL(path.resolve(__dirname,"../studio/index.html")).href);
    await page.locator("#r-spec").setInputFiles(path.join(out,"original_spec.json"));
    await page.locator("#r-field").selectOption(JSON.stringify(["train","lr"]));
    await page.locator("#r-value").fill('"wrong type"');await page.locator("#r-clone").click();
    await page.locator("#r-clone-status").filter({hasText:"Keep the parameter's JSON type"}).waitFor();
    await page.locator("#r-value").fill("0.002");
    const downloadPromise=page.waitForEvent("download");await page.locator("#r-clone").click();const download=await downloadPromise;await download.saveAs(path.join(out,"clone.json"));
    const clone=JSON.parse(fs.readFileSync(path.join(out,"clone.json"),"utf8"));assert.equal(clone.train.lr,.002);assert.notEqual(clone.name,original.name);assert.notEqual(clone.out_dir,original.out_dir);
    const restored=JSON.parse(JSON.stringify(clone));restored.name=original.name;restored.out_dir=original.out_dir;restored.train.lr=original.train.lr;assert.deepEqual(restored,original);
    const jsonl=xs=>Buffer.from(xs.map(x=>JSON.stringify(x)).join("\n"));
    const a=[{event:"start",name:'A <img src=x onerror="throw 1">'},{event:"step",step:1,loss:2},{event:"eval",step:1,val_loss:1.7},{event:"done"}];
    const b=[{event:"start",name:"B clone"},{event:"step",step:1,loss:2.2},{event:"step",step:2,loss:1.8},{event:"eval",step:2,val_loss:1.6},{event:"early_stop",step:2},{event:"done",early_stopped:true}];
    await page.locator("#r-a").setInputFiles({name:"A.jsonl",mimeType:"text/plain",buffer:jsonl(a)});
    await page.locator("#r-b").setInputFiles({name:"B.jsonl",mimeType:"text/plain",buffer:jsonl(b)});
    await page.locator("#r-summary").waitFor({state:"visible"});assert.equal(await page.locator("#r-b-status").textContent(),"early stopped");assert.equal(await page.locator("#r-b-best").textContent(),"1.6000");assert.equal(await page.locator("#research-panel img").count(),0);assert.equal(await page.locator("#r-chart path").count(),2);
    await page.locator("#research-panel").screenshot({path:path.join(out,"research_desktop.png")});
    await page.setViewportSize({width:420,height:1000});await page.waitForTimeout(200);
    const overflow=await page.locator("#research-panel").evaluate(e=>e.scrollWidth>e.clientWidth+2);
    if(overflow){console.log(await page.locator("#research-panel").evaluate(e=>{const r=e.getBoundingClientRect();return [...e.querySelectorAll("*")].map(n=>({tag:n.tagName,id:n.id,right:n.getBoundingClientRect().right,width:n.getBoundingClientRect().width})).filter(n=>n.right>r.right);}));await page.locator("#research-panel").screenshot({path:path.join(out,"research_narrow_failure.png")});}
    assert.equal(overflow,false,"research pane overflow");
    await page.locator("#research-panel").screenshot({path:path.join(out,"research_narrow.png")});
    await page.locator("#r-b").setInputFiles({name:"broken.jsonl",mimeType:"text/plain",buffer:Buffer.from("{bad}")});
    await page.locator("#r-compare-status").filter({hasText:"invalid JSON on line 1"}).waitFor();assert.equal(await page.locator("#r-summary").isVisible(),false);
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,"browser_receipt.json"),JSON.stringify({pass:true,mode:"file-based; installed Chrome headless GPU disabled",checks:["downloaded clone preserves all other fields","bad value type refuses","two synthetic event-log fixtures overlay","early-stop status retained","malformed replacement hides stale comparison","untrusted name stays text","desktop/narrow layout"],page_errors:errors,clone_out_dir:clone.out_dir,chrome:browser.version(),node:process.version,fixture_source:process.env.STUDIO_SPEC || "self-contained synthetic spec",html_sha256:crypto.createHash("sha256").update(fs.readFileSync(path.resolve(__dirname,"../studio/index.html"))).digest("hex"),original_spec_sha256:crypto.createHash("sha256").update(fs.readFileSync(path.join(out,"original_spec.json"))).digest("hex"),clone_sha256:crypto.createHash("sha256").update(fs.readFileSync(path.join(out,"clone.json"))).digest("hex")},null,2)+"\n");
    console.log("Studio browser smoke PASS; clone",path.join(out,"clone.json"));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});