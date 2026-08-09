# dalang.io — GPU Hardware & Investment Plan

_How to grow into AI/GPU hosting without lighting money on fire. Framed for a lean VPS business that has already proven it can run a 33B model on an 8 GB card._

> **Core principle:** buy hardware for what customers **pay for**, not for the biggest model that exists. Rent → prove demand → buy. Never buy ahead of utilization.

---

## 0. The proof we already have (the moat)

We ran **MiniMax-H3 (33B video+audio model)** on a **single RTX 3060 Ti (8 GB)** — hardware the model "officially" needs an 80 GB datacenter GPU for. We profiled the bottleneck to the exact allocation and patched it for a bit-identical ~2 GB VRAM win, raising the native resolution ceiling 1.87×.

**Why this matters commercially:** it's evidence we extract more usable output per dollar of hardware than a typical host. That is the margin story an investor buys.

---

## 1. Do NOT buy for the biggest model

MiniMax-M3 (428B params) needs **~400–900 GB of VRAM** = a **multi-GPU datacenter node** (~8× H100, $250k+ to buy or ~$15–30/hr to rent). Buying that to run a model **no customer is paying for yet** is how infra startups die.

- Curious about M3? → **Use MiniMax's hosted API.** ~$0 capex.
- M3 *understands* media; it doesn't *generate* it. For making videos, **H3 stays the right tool.**

---

## 2. The GPU buying ladder

Each rung is a real business step. Prices are approximate (2026, USD) and move fast.

| Tier | Card | VRAM | ~Price | What it unlocks (paying workloads) |
|---|---|---|---|---|
| **Now (owned)** | RTX 3060 Ti | 8 GB | — | Proof + demos. Thin margins, but the story. |
| **Step 1** ⭐ | **RTX 4090 / 5090** | 24–32 GB | **$2–3k** | Native 720p+ video, 7–13B LLMs, image gen at scale. Best $/revenue. |
| Step 2 | RTX 6000 Ada / A6000 | 48 GB | $5–7k | Bigger models, 2 tenants/card, H3 with no tricks needed. |
| Step 3 | L40S | 48 GB | $8–10k | Datacenter-grade, proper multi-tenant, warranty/ECC. |
| Step 4 | A100 / H100 | 80 GB | $15–30k | "Serious AI host" tier. Only when demand is proven. |
| (Avoid early) | 8× H100 node | 640 GB | $250k+ | Frontier/428B models. Rent this, don't buy. |

**Recommendation: buy ONE RTX 4090/5090 first (~$2–3k).** 3–4× the VRAM of what we run now, single card, plugs into a box we already operate, and it covers ~everything customers will actually ask for.

---

## 3. Buy vs Rent — the numbers

**Rent (RunPod / Vast / Lambda), approximate:**

| Card | Rent $/hr |
|---|---|
| RTX 4090 | $0.30–0.50 |
| A100 80GB | $1.50–2.00 |
| H100 80GB | $2.00–4.00 |

**Payback math for buying (example: one RTX 4090 @ $2,000):**

```
Resell to customers at $0.40/hr
Run 12 hr/day utilized → $0.40 × 12 × 30 ≈ $144/month
Payback ≈ $2,000 / $144 ≈ 14 months
After that → mostly margin (minus power + hosting)
```

**The rule:** a GPU pays off once it runs **> ~50% utilized**. Below that, renting is cheaper and you carry no depreciation.

**Depreciation reality:** GPUs lose ~30%/yr in value and newer/cheaper ones ship constantly. An idle owned GPU is money burning. This is *the* argument for rent-until-proven.

---

## 4. The staged plan (capital-efficient)

```
Phase 0  (now)      : 8 GB box. Demos, marketing, proof. ~$0 new capex.
Phase 1  (rent)     : Rent 4090/A100 hourly. Serve first paying AI customers.
                      Measure: which workloads sell? what utilization?
Phase 2  (buy 1)    : Utilization > 50% on rented cards → buy ONE 4090/5090 (~$2-3k).
Phase 3  (buy 2-3)  : First card > 50% utilized → buy 2-3 more. Rent for spikes.
Phase 4  (scale)    : Sustained demand for big models → L40S/A100. Still rent the frontier.
```

Each phase is only entered **after the previous one's utilization justifies it.** Demand pulls hardware; hardware never pushed ahead of demand.

---

## 5. Investor one-pager (send this)

> **dalang.io — GPU hosting, the capital-efficient way**
>
> **What we proved:** we ran a 33-billion-parameter AI video model on an **8 GB gaming GPU** — hardware the model officially needs an 80 GB datacenter card for. We profiled the memory bottleneck to the exact kernel and patched it (bit-identical, no quality loss) for a ~2 GB VRAM saving, raising output resolution 1.87×. Translation: **we get more paid output per dollar of silicon than a standard host.**
>
> **The ask:** we are **not** asking to buy a datacenter. We want **[$X]** to (a) buy **2–3 prosumer GPUs (RTX 4090/5090-class, ~$2–3k each)** to serve paying GenAI customers, and (b) a **rental budget** to absorb demand spikes without capex.
>
> **The model:** rent → prove demand → buy only when utilization > 50%. This keeps capex low, avoids the ~30%/yr GPU depreciation trap, and scales spend in lockstep with revenue.
>
> **Why now:** GenAI demand in Indonesia (video, image, LLM) is rising, latency-sensitive customers want local hosting, and we are the cheapest efficient option. The 8 GB build is our de-risked proof.
>
> **Use of funds:** [X% hardware, Y% rental runway, Z% ops]. **Milestone:** [N paying customers / $M MRR] within [months], at which point we buy the next tranche of cards.

---

## 6. TL;DR

- **Try M3:** API, ~$0 hardware.
- **Grow the business:** buy **one RTX 4090/5090 (~$2–3k)** first; rent anything bigger; buy the next card only when the first is >50% utilized.
- **Investor story:** "cheapest hardware, squeezed hardest, scaled by demand — not a pile of depreciating H100s."
- **Never** buy a cluster for a model no one is paying you to run.
