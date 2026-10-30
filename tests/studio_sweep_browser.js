"use strict";
const {chromium}=require("playwright");const fs=require("node:fs");const path=require("node:path");const os=require("node:os");const crypto=require("node:crypto");const assert=require("node:assert/strict");const {pathToFileURL}=require("node:url");
(async()=>{
 const output=path.resolve(process.argv[2] || path.join(os.tmpdir(),"paperkiln-studio-sweep"));fs.mkdirSync(output,{recursive:true});
 const html=path.resolve(__dirname,"../studio/index.html"), real=path.resolve(__dirname,"../experiments/atlas_stage2/cells.jsonl");
 const browser=await chromium.launch({...(process.env.STUDIO_CHROME?{executablePath:process.env.STUDIO_CHROME}:{channel:"chrome"}),headless:true,args:["--disable-gpu","--disable-background-networking","--disable-extensions"]});
 const context=await browser.newContext({viewport:{width:1160,height:1000}}),page=await context.newPage(),errors=[];page.on("pageerror",e=>errors.push(e.message));
 const checks=[];const row=(cell,x,y,extras={})=>({cell,factors:{x,y},n_seeds:3,best_val_mean:2+cell/10,best_val_std:.1,best_val_min:1.9,...extras});
 const upload=async(rows)=>{await page.locator("#s-file").setInputFiles({name:"cells.jsonl",mimeType:"text/plain",buffer:Buffer.from(rows.map(x=>JSON.stringify(x)).join("\n"))});};
 try{
  await page.goto(pathToFileURL(html).href);await page.locator("#s-file").setInputFiles(real);await page.locator("#s-filters select").first().waitFor();assert.equal(await page.locator("#s-filters select").count(),5);await page.locator("#s-grid table").waitFor();checks.push("actual seven-factor mtsweep file loads with five fixed factors and an observed default slice");
  const rows=[row(0,.001,"exact"),row(1,.003,"exact"),row(2,.001,"kimi"),row(3,.003,"kimi",{n_seeds:0,best_val_mean:null,best_val_std:null})];
  await upload(rows);await page.locator("#s-grid td").first().waitFor();assert.equal(await page.locator("#s-grid td").count(),4);assert.match(await page.locator("#s-grid").innerText(),/2.0000/);assert.match(await page.locator("#s-grid").innerText(),/no recorded value/);checks.push("saved means/n/population SD and zero-seed cells rendered");
  await page.locator("#sweep-panel").screenshot({path:path.join(output,"heatmap_desktop.png")});
  await page.setViewportSize({width:390,height:1000});await page.locator("#sweep-panel").screenshot({path:path.join(output,"heatmap_narrow.png")});
  const overflow=await page.locator("#sweep-panel").evaluate(e=>({width:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right,viewport:innerWidth}));assert.ok(overflow.right<=overflow.viewport+1);checks.push("narrow layout keeps horizontal grid scroll within panel");
  await upload([row(0,0,false),row(1,0,false),row(2,1,true)]);await page.locator("#s-grid").getByText("multiple cells (2)",{exact:true}).waitFor();assert.match(await page.locator("#s-grid").innerText(),/unobserved/);checks.push("ambiguous coordinates remain separate and unobserved gaps explicit");
  await upload([row(0,'<img src=x onerror="throw 1">',1)]);await page.locator("#s-status").filter({hasText:"1 recorded cells"}).waitFor();assert.ok((await page.locator("#s-grid").innerText()).includes(JSON.stringify('<img src=x onerror="throw 1">')));assert.equal(await page.locator("#s-grid img").count(),0);checks.push("untrusted axis labels remain text");
  await upload([row(0,1,2),row(0,2,3)]);await page.locator("#s-status").filter({hasText:"Duplicate cell"}).waitFor();assert.equal(await page.locator("#s-grid table").count(),0);assert.equal(await page.locator("#s-x").isDisabled(),true);checks.push("malformed replacement hides old grid and disables controls");
  assert.deepEqual(errors,[]);const sha=p=>crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
  fs.writeFileSync(path.join(output,"browser_receipt.json"),JSON.stringify({pass:true,checks,page_errors:errors,chrome:browser.version(),node:process.version,html_sha256:sha(html),real_fixture:{path:real,sha256:sha(real)},scope:"file-based installed Chrome; no GPU/model/training/network calls by heatmap"},null,2)+"\n");console.log(JSON.stringify({pass:true,checks}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});