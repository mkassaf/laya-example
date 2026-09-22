# laya example

A minimal, ready-to-run example for [**laya**](https://github.com/NandhaKishorM/laya) —
a fast, non-autoregressive "System 1" decision engine from Convai Innovations.

## What laya is

Most LLM pipelines answer structured questions ("which department?", "how
urgent?", "is this a refund request?") by generating text and parsing it back
out. That's slow, and an autoregressive model can hallucinate a label that was
never in your schema.

laya skips generation entirely. You give it a **state** (an email, a ticket, a
JSON document — in any of 100+ languages) and a set of **typed questions**,
and it answers all of them in a single forward pass through a BERT-family
encoder:

| Type | Returns | Use for |
|---|---|---|
| `choice` | top label + full probability distribution + confidence | routing, intent, category |
| `score` | expected value on an ordinal rubric | urgency, severity, frustration |
| `noul` | calibrated P(true), 0.0–1.0 | churn risk, spam, jailbreak/phishing detection |

Because it's not sampling tokens, there's nothing to parse and nothing to
hallucinate — the output is a typed, structured decision every time.

## Why it's good

- **Fast.** ~33 ms for one question, ~7 ms/question batched, measured on a T4
  GPU. No autoregressive decoding loop.
- **Calibrated on purpose.** Trained with reinforcement learning against
  strictly proper scoring rules (RLCD), so a 0.86 confidence is meant to
  actually mean 86% — not just a token logit dressed up as a probability.
  That makes automated confidence gating (auto-act above a threshold,
  escalate to a human below it) a reasonable thing to build on top of it.
- **Actually multilingual, and honest about the limits.** The English
  checkpoint collapses on non-Latin scripts while staying *confidently*
  wrong (0.000 accuracy at 95% confidence on Khmer, per the project's own
  benchmarks) — so a built-in `Router` detects script/language in under
  0.5 ms and picks the right checkpoint before the forward pass even runs,
  rather than relying on the model to know what it doesn't know.
- **Cheap to self-host.** Apache 2.0 weights, runs on CPU or GPU, no
  per-token API cost. The project's own benchmarks show it several times
  faster than a comparable closed-API alternative on structured decisions,
  though it trails on very high-cardinality label sets (50+ options) — see
  [`BENCHMARKS.md` upstream](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
  for the full, measured comparison and honest caveats.
- **Small and specializable.** The base checkpoints are near chance
  zero-shot on hard, fine-grained tasks — the real value shows up after a
  few hours of fine-tuning on your own labeled data (a provided notebook
  runs the whole RLCD fine-tuning loop on free Kaggle GPUs).

## How it works, briefly

`laya` ships three checkpoints and a `Router` that picks between them per
request:

- `laya` — ModernBERT-large, English-tuned
- `laya-multilingual` — mmBERT-base, 100+ languages, 2x faster
- `laya-typed-decisions` — ModernBERT-large, fine-tuned for typed-decision
  workflows

`Router(preload=True)` loads all checkpoints into memory up front so a
language switch doesn't force a multi-second reload; `router.predict(state,
questions)` returns both the answers and the routing decision (which model,
and why).

## This repo

```
requirements.txt          # pip install -r requirements.txt -> laya
examples/ticket_triage.py # runnable example: department/urgency/churn/refund
                           # on an English and a Hindi support ticket
```

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python examples/ticket_triage.py
```

First run downloads the `convaiinnovations/laya` and
`convaiinnovations/laya-multilingual` checkpoints from Hugging Face Hub (a
few hundred MB each) and caches them under `~/.cache/huggingface`. Needs a
network connection for that first run; works on CPU (slower) or CUDA
(faster) after that.

**Verified output** (English ticket routed to the `english` checkpoint,
Hindi ticket auto-routed to `multilingual`):

```
Routing model : english
Routing reason: English Latin text

Department       : billing (confidence: 0.86)
Urgency          : 1.44
Churn risk       : 0.825
Refund requested : 0.843

Hindi ticket routed to: multilingual
Department             : billing
```

## Credit

All model weights, training, and benchmarks are from
[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache 2.0,
Convai Innovations). This repo is just a runnable example on top of it.
