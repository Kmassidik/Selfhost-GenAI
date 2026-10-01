/* Local renderer for this mini-site. Render logic (currentFile, renderRawMath,
   renderArticle, initMermaid, DOMContentLoaded) is copied verbatim from
   knowledge-base/assets/app.js. Only the nav data + buildSidebar branding differ:
   a FLAT list of PAGES, no categories. */
const PAGES = [
  { file:"index.html", n:"·", title:"Overview" },
  { file:"QUEUE.html", n:"★", title:"Build Queue (live)" },
  { file:"PROJECT-SUMMARY.html", n:"01", title:"Project Summary" },
  { file:"CHECKPOINT-v0.1.html", n:"02", title:"Checkpoint v0.1" },
  { file:"SETUP.html", n:"03", title:"Setup" },
  { file:"hardware-investment-plan.html", n:"04", title:"Hardware & Investment Plan" },
  { file:"runpod-plan.html", n:"05", title:"RunPod H100 Plan" },
  { file:"BOX-CLEANUP-PLAN.html", n:"06", title:"Box Folder Cleanup Plan" },
];

function buildSidebar(){
  const cur = currentFile();
  const link = p => `<a class="${p.file===cur?"active":""}" href="${p.file}">
        <span class="n">${p.n}</span><span>${p.title}</span></a>`;
  const nav = document.createElement("nav");
  nav.className = "sidebar";
  nav.innerHTML = `
    <a class="brand" href="index.html" style="text-decoration:none;color:inherit">
      <span class="dot"></span>
      <span><b>🗂️ Planning</b><small>plans · checkpoints · decisions</small></span>
    </a>
    <div class="nav">
      ${PAGES.map(link).join("")}
    </div>
    <div class="foot">
      <a href="../index.html" style="color:#5ee0c0;text-decoration:none">← Vault hub</a>
    </div>`;
  document.body.prepend(nav);
}

function currentFile(){
  const p = location.pathname.split("/").pop();
  return p && p.length ? p : "index.html";
}

/* marked passes raw HTML blocks (our callout <div>s) straight through, so the math
   extension never sees $…$ inside them. This second pass walks the rendered DOM and
   renders any leftover $…$ / $$…$$ in text nodes, skipping code and already-rendered
   KaTeX. Result: math renders everywhere, including inside callouts. */
function renderRawMath(root){
  if(!window.katex) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n){
      if(!n.nodeValue.includes("$")) return NodeFilter.FILTER_REJECT;
      if(n.parentElement && n.parentElement.closest("pre,code,.katex,.katex-display")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const targets=[]; let n;
  while((n=walker.nextNode())) targets.push(n);
  for(const t of targets){
    const out = t.nodeValue
      .replace(/\$\$([\s\S]+?)\$\$/g, (_,m)=>katex.renderToString(m.trim(),{displayMode:true,throwOnError:false}))
      .replace(/\$([^$\n]+?)\$/g,     (_,m)=>katex.renderToString(m.trim(),{displayMode:false,throwOnError:false}));
    if(out !== t.nodeValue){
      const span = document.createElement("span");
      span.innerHTML = out;
      t.replaceWith(span);
    }
  }
}

function renderArticle(){
  const md = document.getElementById("md").textContent;
  const wrap = document.createElement("div");
  wrap.className = "wrap";
  const art = document.createElement("article");
  art.className = "article";
  marked.setOptions({ gfm:true, breaks:false });
  // LaTeX math via KaTeX: $$...$$ block and $...$ inline
  if(window.katex){
    marked.use({ extensions:[
      { name:"math", level:"block",
        start:(s)=>s.match(/\$\$/)?.index,
        tokenizer:(s)=>{ const c=/^\$\$([\s\S]+?)\$\$/.exec(s); return c?{type:"math",raw:c[0],math:c[1].trim()}:undefined; },
        renderer:(t)=>`<div class="katex-display">${katex.renderToString(t.math,{displayMode:true,throwOnError:false})}</div>`
      },
      { name:"inlineMath", level:"inline",
        start:(s)=>s.match(/\$/)?.index,
        tokenizer:(s)=>{ const c=/^\$([^$\n]+?)\$/.exec(s); return c?{type:"inlineMath",raw:c[0],math:c[1].trim()}:undefined; },
        renderer:(t)=>katex.renderToString(t.math,{displayMode:false,throwOnError:false})
      }
    ]});
  }
  art.innerHTML = marked.parse(md);
  renderRawMath(art);   // catch $…$ that marked left raw inside HTML blocks (e.g. callouts)
  wrap.appendChild(art);

  // promote mermaid code blocks
  art.querySelectorAll("code.language-mermaid").forEach(code=>{
    const div = document.createElement("div");
    div.className = "mermaid";
    div.textContent = code.textContent;
    code.closest("pre").replaceWith(div);
  });

  // prev / next pager
  const cur = currentFile();
  const idx = PAGES.findIndex(p=>p.file===cur);
  const prev = idx>0 ? PAGES[idx-1] : null;
  const next = idx>=0 && idx<PAGES.length-1 ? PAGES[idx+1] : null;
  const pager = document.createElement("div");
  pager.className = "pager";
  pager.innerHTML =
    (prev?`<a href="${prev.file}"><span class="dir">← Previous</span><span class="ttl">${prev.title}</span></a>`:`<span style="flex:1"></span>`) +
    (next?`<a class="next" href="${next.file}"><span class="dir">Next →</span><span class="ttl">${next.title}</span></a>`:`<span style="flex:1"></span>`);
  art.appendChild(pager);

  document.body.appendChild(wrap);
}

function initMermaid(){
  if(!window.mermaid) return;
  mermaid.initialize({
    startOnLoad:false, securityLevel:"loose", theme:"base",
    themeVariables:{
      darkMode:true, background:"#12161f",
      primaryColor:"#151b26", primaryTextColor:"#e7ecf3", primaryBorderColor:"#2f3b4f",
      lineColor:"#7aa2ff", secondaryColor:"#122", tertiaryColor:"#0d1017",
      fontFamily:"Inter, sans-serif", fontSize:"14px",
      clusterBkg:"#0f141d", clusterBorder:"#2a3446",
      nodeBorder:"#3a4a63", edgeLabelBackground:"#0d1017",
    }
  });
  try { mermaid.run({ querySelector:".mermaid" }); }
  catch(e){ console.error("mermaid", e); }
}

document.addEventListener("DOMContentLoaded", ()=>{
  buildSidebar();
  renderArticle();
  initMermaid();
});
