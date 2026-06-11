# Epiphany — Pitch & Demo Script

**Live demo:** https://epiphany-ds.fly.dev/ · **Repo:** (your GitHub URL)
**Built for:** Google Cloud Rapid Agent Hackathon

> ⚠️ Verify against the hackathon site: required submission format (video length,
> deck, public repo), judging criteria, and any "must use" Google Cloud product
> tags. Tailor the emphasis below to whatever the rubric weights most.

---

## 1. The 30-second hook (memorize this)

> "Every company is sitting on data they never analyze, because hiring a data
> scientist is slow and expensive. **Epiphany is an autonomous AI data scientist
> that works while you sleep.** Point it at *any* dataset and it explores the
> data, forms a real hypothesis, **proves it with the right statistical test**,
> trains a real model, and ships it — completely on its own. It's not a chatbot
> that talks about data. It *does* the data science."

---

## 2. The problem (20s)

- Data sits unused: 73% of enterprise data is never analyzed.
- Data scientists are scarce, expensive, and slow — every question is a ticket
  in a queue.
- "AI" tools today mostly *describe* data or generate code you still have to run
  and trust. Nobody closes the loop from raw data → validated insight → deployed
  model.

## 3. The solution (30s)

Epiphany is an **autonomous agent** that runs a continuous 5-step loop modeled on
how a real data scientist works — and every step is **real**, not simulated:

| Step | Powered by | What actually happens |
|------|-----------|------------------------|
| 1. Trigger | **Fivetran** | Wakes when new data syncs |
| 2. Explore | **Elastic** | Discovers the real schema, ranks real signals |
| 3. Reason | **Gemini + Google ADK** | An LLM agent forms a falsifiable hypothesis |
| 4. Validate | **Python sandbox (SciPy)** | Runs the *right* test — χ²/t-test/ANOVA/correlation — on real rows |
| 5. Deploy | **GitLab** | Trains a real scikit-learn model, opens a Merge Request |

## 4. Why it wins — the one-liner differentiators (45s)

Say these out loud; they're what separates you from the pack:

1. **It's real, not a demo prop.** The statistics are computed with SciPy on real
   data. A weak relationship comes back *not significant* — the system can say
   "no," which fake demos never can. The model is actually trained and scored
   (real ROC-AUC / R²), and saved as a loadable `.pkl`.
2. **It works on ANY dataset.** Upload a CSV in the browser and it auto-detects
   column types, picks a target, and chooses the correct test. We demo it live on
   three different domains — customer churn, wine chemistry, diabetes — and it
   adapts the test *and* the model each time. No hardcoding.
3. **It's genuinely autonomous.** A background loop runs forever, rotating across
   targets and angles. The agent dynamically decides which tools to call via the
   **Google Agent Development Kit** — this is agentic, not a fixed script.
4. **It's safe.** Agent-generated code is screened by an AST security scanner and
   executed in a network-isolated, resource-limited sandbox.
5. **It's deployed and usable.** Live on the web, real auth (Clerk), connect your
   own providers from the UI, graceful degradation so it always works.

## 5. Google Cloud story (20s — emphasize if the rubric weights it)

- **Gemini 2.5 Flash** is the reasoning brain.
- **Google Agent Development Kit (ADK)** orchestrates the autonomous tool-calling.
- Container is **Cloud Run-ready** (Dockerfile, `$PORT`, health checks).
- The MCP integrations (Fivetran, Elastic, GitLab) mirror the agentic-data stack.

---

## 6. Live demo walkthrough (90 seconds — the money shot)

Have https://epiphany-ds.fly.dev/ open and signed in. Have 1–2 CSVs ready (ideally
of different shapes — e.g. one with a category target, one with a numeric target).

1. **Landing → sign in.** "Real auth via Clerk — anyone can sign up." (5s)
2. **Empty workspace.** "Notice — it starts empty. No canned data. It's *your*
   data scientist; you bring the data." (8s)
3. **Upload a dataset.** Click **Upload data**, drop in your first CSV. "It
   auto-detects the columns and the target and starts working immediately — the
   **Active Agent Stream** is the agent running its loop live." (20s)
4. **The proof it's real — upload a different dataset.** Drop in a second CSV
   with a different target type. "Watch — it re-profiles and now chooses a
   **different statistical test** (t-test → ANOVA → correlation) and a matching
   model. Same agent, totally different data, zero config." (20s)
5. **Run a mission.** Type a question into Mission Control in plain English →
   **Deploy Agent**. "Now Gemini + the Google ADK take over and dynamically
   choose which tools to call." Show the stream + the generated model code
   (syntax-highlighted) + the trained-model metrics. (22s)
6. **Hero metrics + Interventions table.** "Every row is a real hypothesis it
   tested, with a real p-value. Green = statistically significant." (10s)

**Closing line:**
> "Epiphany turns a dataset into a validated, deployed model without a human in
> the loop — and it just did it live, on data it had never seen, in 90 seconds.
> That's not a demo of data science. That *is* data science, automated."

---

## 7. Likely judge questions (have answers ready)

- **"Is the analysis actually real or scripted?"** → Real SciPy tests on real
  rows; show the correlation case returning *not significant*. Open a saved
  findings report in `reports/`.
- **"What if the LLM hallucinates a column?"** → The hypothesis is validated
  against the real schema; hallucinated columns are rejected and repaired from
  the measured association ranking.
- **"Isn't running generated code dangerous?"** → AST denylist + network-isolated
  subprocess with CPU/memory limits. Show `scripts/test_sandbox_security.py`.
- **"Does it scale beyond the demo data?"** → Live path reads from Elastic; the
  same code runs on a local file when Elastic isn't configured.
- **"What's the business model / who's it for?"** → Analysts and PMs at any
  company with data and no data-science bandwidth; self-serve, bring-your-own-data.

## 8. 60-second elevator version (if time is tight)

> "Epiphany is an autonomous AI data scientist. Give it any dataset and it
> explores it, forms a hypothesis, proves it with the correct statistical test,
> trains a real model, and opens a merge request — by itself, in a loop, in the
> background. Built on Gemini and the Google Agent Development Kit. It's live on
> the web right now, it works on any domain — churn, healthcare, anything — and
> unlike most 'AI data' tools, every number is real: it runs real SciPy tests and
> can tell you when there's *no* signal. It's data science that runs itself."
