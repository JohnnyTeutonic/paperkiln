"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const html=fs.readFileSync(path.join(__dirname,"../studio/index.html"),"utf8");
const core=html.split("// BEGIN STUDIO_EXTRACTION_CORE")[1].split("// END STUDIO_EXTRACTION_CORE")[0];
function fixture(fetch){
  class Element {
    constructor(){this.children=[];this.value="";this.textContent="";this.style={};this.disabled=false;this.hidden=true;}
    appendChild(child){this.children.push(child);return child;}
    append(...children){this.children.push(...children);}
    replaceChildren(...children){this.children=[...children];}
    set innerHTML(value){throw Error("Untrusted extraction must not be inserted as HTML");}
  }
  const elements={};const $=id=>elements[id] ||= new Element();
  const context={fetchTimer:null,module:{exports:{}},document:{createElement:()=>new Element()},$,fetch,
    setInterval:()=>1,clearInterval:()=>{},Number,JSON,Error};
  vm.runInNewContext(core+"\nmodule.exports={reviewExtraction,pollExtraction,doLlmExtract};",context);
  return {...context.module.exports,$,elements};
}
const result={title:'<img src=x onerror="bad()">',arxiv_id:"1234.56789",fields:{norm:{value:"rmsnorm",verdict:"unverified",evidence:'<script>bad()</script>'}},unresolved:["n_layers"],mechanism_scan:[{blocking:true,mechanism:"novel"}]};
test("actual verification/unresolved/blockers render as text with no builder application",()=>{
  const f=fixture(()=>{throw Error("no requests");});f.reviewExtraction(result);
  const serialized=JSON.stringify(f.$("p-result"));
  assert.ok(serialized.includes("unverified"));assert.ok(serialized.includes("n_layers"));assert.ok(serialized.includes("blocking"));assert.ok(serialized.includes("<img"));
  assert.deepEqual(Object.keys(f.elements),["p-result"]); // never touches builder dimensions or training controls
});
test("completed result is fetched once per job; running/cancelled status changes controls",async()=>{
  let state="extracting",results=0;
  const f=fixture(async url=>({ok:true,json:async()=>url==="extractstatus"?{state,job:"job-1",configured:true}:(results++,result)}));
  await f.pollExtraction();assert.equal(f.$("p-go").disabled,true);assert.equal(f.$("p-cancel").hidden,false);
  state="done";await f.pollExtraction();await f.pollExtraction();assert.equal(results,1);
  state="cancelled";await f.pollExtraction();assert.equal(f.$("p-go").disabled,false);assert.equal(f.$("p-cancel").hidden,true);
});
test("failed LLM request reports failure without rule-based fallback",async()=>{
  const calls=[];const f=fixture(async(url,options)=>{calls.push(url);return{ok:false,text:async()=>"missing configured key"};});
  f.$("p-id").value="1234.56789";f.$("p-budget").value="1";await f.doLlmExtract();
  assert.deepEqual(calls,["extract"]);assert.equal(f.$("p-status").textContent,"missing configured key");
});
