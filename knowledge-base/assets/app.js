/* Shared renderer for the knowledge base.
   Each page embeds Markdown in <script type="text/markdown" id="md">.
   We render it with marked, promote ```mermaid blocks to Mermaid diagrams,
   build the sidebar nav, and add prev/next paging. Fully offline. */

const PAGES = [
  { file: "index.html",              n: "00", title: "Start Here" },
  { file: "01-what-is-genai.html",   n: "01", title: "What Is Generative AI" },
  { file: "02-your-hardware.html",   n: "02", title: "Your Hardware & the Offload Trick" },
  { file: "03-how-h3-works.html",    n: "03", title: "How MiniMax-H3 Works" },
  { file: "04-software-stack.html",  n: "04", title: "The Software Stack" },
  { file: "05-our-build-story.html", n: "05", title: "Our Build — The True Story" },
  { file: "07-optimization-720p.html",n: "07", title: "Optimizing for 720p" },
  { file: "08-how-a-prompt-becomes-video.html",n: "08", title: "How a Prompt Becomes a Video" },
  { file: "09-from-2017-cv-to-genai.html",n: "09", title: "From 2017 CV to Modern GenAI" },
  { file: "10-the-8gb-experiment.html",n: "10", title: "The 8 GB Experiment" },
  { file: "06-learning-roadmap.html",n: "06", title: "Your Learning Roadmap" },
];

function currentFile(){
  const p = location.pathname.split("/").pop();
  return p && p.length ? p : "index.html";
}

function buildSidebar(){
  const cur = currentFile();
  const nav = document.createElement("nav");
  nav.className = "sidebar";
  nav.innerHTML = `
    <a class="brand" href="index.html" style="text-decoration:none;color:inherit">
      <span class="dot"></span>
      <span><b>Self-Hosting GenAI</b><small>Field guide &amp; roadmap</small></span>
    </a>
    <div class="nav">
      ${PAGES.map(p => `<a class="${p.file===cur?"active":""}" href="${p.file}">
        <span class="n">${p.n}</span><span>${p.title}</span></a>`).join("")}
    </div>
    <div class="foot">Built from a real build on an<br>RTX 3060 Ti (8&nbsp;GB) box.<br>Every spec here is measured, not rounded.</div>`;
  document.body.prepend(nav);
}

function renderArticle(){
  const md = document.getElementById("md").textContent;
  const wrap = document.createElement("div");
  wrap.className = "wrap";
  const art = document.createElement("article");
  art.className = "article";
  marked.setOptions({ gfm:true, breaks:false });
  art.innerHTML = marked.parse(md);
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
