/* Shared renderer for the knowledge base.
   Each page embeds Markdown in <script type="text/markdown" id="md">.
   We render it with marked, promote ```mermaid blocks to Mermaid diagrams,
   build the sidebar nav, and add prev/next paging. Fully offline. */

/* Sidebar is grouped by the NATURE of each chapter, not a flat 01–35 list.
   The build-story / diary chapters (05, 10) live in the Journey site, not here. */
const CATEGORIES = [
  { key: "foundations", label: "Foundations",         icon: "🌱" },
  { key: "technical",   label: "Technical Deep-Dives", icon: "⚙️" },
  { key: "ideas",       label: "Ideas & Strategy",     icon: "🧠" },
  { key: "glossary",    label: "Glossary",             icon: "📖" },
];
const PAGES = [
  { file: "index.html",              n: "00",  title: "Start Here", cat: "home" },

  // 🌱 Foundations — the concepts, what things actually are
  { file: "01-what-is-genai.html",   n: "01",  title: "What Is Generative AI", cat: "foundations" },
  { file: "02-your-hardware.html",   n: "02",  title: "Your Hardware & the Offload Trick", cat: "foundations" },
  { file: "03-how-h3-works.html",    n: "03",  title: "How MiniMax-H3 Works", cat: "foundations" },
  { file: "04-software-stack.html",  n: "04",  title: "The Software Stack", cat: "foundations" },
  { file: "08-how-a-prompt-becomes-video.html", n: "08", title: "How a Prompt Becomes a Video", cat: "foundations" },
  { file: "09-from-2017-cv-to-genai.html", n: "09", title: "From 2017 CV to Modern GenAI", cat: "foundations" },
  { file: "19-what-a-model-is.html", n: "19",  title: "What a Model Actually Is", cat: "foundations" },
  { file: "28-anatomy-of-h3.html",   n: "28",  title: "Anatomy of H3", cat: "foundations" },
  { file: "06-learning-roadmap.html",n: "06",  title: "Your Learning Roadmap", cat: "foundations" },
  { file: "50-reading-an-image-recipe.html", n: "50", title: "Reading an Image's Recipe", cat: "foundations" },

  // ⚙️ Technical — how it actually works under the hood
  { file: "07-optimization-720p.html", n: "07", title: "Optimizing for 720p", cat: "technical" },
  { file: "13-int8-vs-q4.html",      n: "13",  title: "int8 vs Q4 on This Box", cat: "technical" },
  { file: "14-the-math-behind-it.html", n: "14", title: "The Math Behind It All", cat: "technical" },
  { file: "15-vectors-matrices-signals.html", n: "15", title: "Vectors, Matrices, or Signals?", cat: "technical" },
  { file: "16-multi-gpu-and-nvlink.html", n: "16", title: "Multi-GPU & NVLink", cat: "technical" },
  { file: "18-number-formats-precision.html", n: "18", title: "Number Formats & Precision", cat: "technical" },
  { file: "22-pipeline-in-detail.html", n: "22", title: "The Pipeline in Full Detail", cat: "technical" },
  { file: "23-math-from-scratch.html", n: "23", title: "The Math, From Scratch", cat: "technical" },
  { file: "24-pcie-nvlink-interconnect.html", n: "24", title: "PCIe & NVLink — The Interconnect", cat: "technical" },
  { file: "29-where-numbers-live.html", n: "29", title: "Where the Numbers Live", cat: "technical" },
  { file: "30-the-render-live.html", n: "30",  title: "The Render, Live", cat: "technical" },
  { file: "31-what-the-gpu-does.html", n: "31", title: "What the GPU Is Actually Doing", cat: "technical" },
  { file: "32-thinking-like-a-circuit.html", n: "32", title: "Thinking Like a Circuit", cat: "technical" },
  { file: "34-open-use-close-flashattention.html", n: "34", title: "Open, Use, Close: FlashAttention", cat: "technical" },
  { file: "40-lora-how-it-works.html", n: "40", title: "LoRA — A Frozen Giant, a Tiny Nudge", cat: "technical" },
  { file: "41-python-is-the-glue.html", n: "41", title: "Python Is the Glue, CUDA Is the Engine", cat: "technical" },
  { file: "42-why-two-models.html", n: "42", title: "Why Two Models? Sharing the Recipe", cat: "technical" },
  { file: "43-the-dalang-engine.html", n: "43", title: "The Dalang Engine", cat: "ideas" },
  { file: "44-building-dalang.html", n: "44", title: "Building Dalang — the Log", cat: "ideas" },
  { file: "45-three-gpu-sequence-parallelism.html", n: "45", title: "Three GPUs, One Sequence", cat: "ideas" },
  { file: "46-the-first-15-seconds.html", n: "46", title: "The First 15 Seconds", cat: "ideas" },
  { file: "47-the-scorecard.html", n: "47", title: "The Scorecard", cat: "ideas" },
  { file: "48-the-pipeline-without-comfyui.html", n: "48", title: "The Pipeline, Without ComfyUI", cat: "technical" },

  // 🧠 Ideas & Strategy — where ideas come from, the calls, the frontier
  { file: "12-pushing-past-8gb.html", n: "12", title: "Pushing Past 8 GB", cat: "ideas" },
  { file: "17-what-actually-improves-quality.html", n: "17", title: "What Actually Improves Quality", cat: "ideas" },
  { file: "20-how-far-we-can-go.html", n: "20", title: "How Far We Can Go", cat: "ideas" },
  { file: "21-build-vs-buy.html",    n: "21",  title: "Can We Beat the Hardware by Building?", cat: "ideas" },
  { file: "25-inference-engine-landscape.html", n: "25", title: "The Inference Engine Landscape", cat: "ideas" },
  { file: "26-serving-h3-real-world.html", n: "26", title: "Serving H3 in the Real World", cat: "ideas" },
  { file: "27-hardware-budget-audit.html", n: "27", title: "Hardware & Budget Audit", cat: "ideas" },
  { file: "33-long-video-small-card.html", n: "33", title: "Long Video on a Small Card", cat: "ideas" },
  { file: "35-is-it-really-ours.html", n: "35", title: "Is It Really Ours?", cat: "ideas" },
  { file: "36-splitting-the-video-cp.html", n: "36", title: "Splitting the Video — Context Parallelism", cat: "ideas" },
  { file: "37-the-production-line.html", n: "37", title: "The Production Line", cat: "ideas" },
  { file: "38-making-the-movie.html", n: "38", title: "Making the Movie", cat: "ideas" },
  { file: "39-choosing-the-next-card.html", n: "39", title: "Choosing the Next Card", cat: "ideas" },
  { file: "49-the-image-engine.html", n: "49", title: "The Image Engine — Qwen-Image 2.1", cat: "ideas" },
  { file: "51-the-benchmark.html", n: "51", title: "The Benchmark — 4 Models on 8 GB", cat: "ideas" },

  // 📖 Glossary
  { file: "11-glossary.html",        n: "11",  title: "Glossary — Every Term", cat: "glossary" },
  { file: "glossary.html",           n: "A–Z", title: "Master Glossary", cat: "glossary" },
];

function currentFile(){
  const p = location.pathname.split("/").pop();
  return p && p.length ? p : "index.html";
}

function buildSidebar(){
  const cur = currentFile();
  const home = PAGES.find(p => p.cat === "home");
  const groups = CATEGORIES
    .map(c => ({ ...c, items: PAGES.filter(p => p.cat === c.key) }))
    .filter(g => g.items.length);

  const link = p => `<a class="${p.file===cur?"active":""}" href="${p.file}">
        <span class="n">${p.n}</span><span>${p.title}</span></a>`;
  const header = g => `<div class="cat-h" style="margin:16px 4px 5px;font-size:11px;`
    + `letter-spacing:.11em;text-transform:uppercase;color:#7c8aa0;font-weight:700">`
    + `${g.icon} ${g.label}</div>`;

  const nav = document.createElement("nav");
  nav.className = "sidebar";
  nav.innerHTML = `
    <a class="brand" href="index.html" style="text-decoration:none;color:inherit">
      <span class="dot"></span>
      <span><b>Self-Hosting GenAI</b><small>Knowledge base</small></span>
    </a>
    <div class="nav">
      ${home ? link(home) : ""}
      ${groups.map(g => header(g) + g.items.map(link).join("")).join("")}
    </div>
    <div class="foot">
      <a href="../index.html" style="color:#5ee0c0;text-decoration:none">← Vault hub</a><br><br>
      Built from a real build on a<br>3× RTX 3060 Ti (8&nbsp;GB) box.<br>Every spec measured, not rounded.</div>`;
  document.body.prepend(nav);
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
