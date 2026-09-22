"""
Minimal runnable example for laya (https://github.com/NandhaKishorM/laya).

Routes a support ticket to a department, scores its urgency, and flags
churn risk / refund requests -- all in a single forward pass per checkpoint.

Run:
    pip install -r requirements.txt
    python examples/ticket_triage.py
"""

import os

# Once a checkpoint is cached locally (~/.cache/huggingface/hub), huggingface_hub still
# does an online cache-verification round trip on every call by default, which is what
# prints "Fetching N files" each run even though nothing is re-downloaded. Going offline
# skips that network check entirely. If a checkpoint isn't cached yet (fresh checkout),
# offline mode raises, so fall back to a normal online run, which downloads and caches it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from laya import Router

# preload=True (with no names) would build all three bundled checkpoints, including
# typed-decisions, which this example never routes to. Preloading just the two checkpoints
# actually used here (routed by script/language -- see laya/router.py) skips that wasted
# build every run.
try:
    router = Router()
    router.preload(["english", "multilingual"])
except Exception:
    os.environ.pop("HF_HUB_OFFLINE", None)
    router = Router()
    router.preload(["english", "multilingual"])

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": (
        "Hi, we were billed twice for March. Please refund the duplicate "
        "today or we will cancel our plan."
    ),
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

result = router.predict(state, questions)
answers = result["answers"]

print("Routing model :", result["routing"]["model"])
print("Routing reason:", result["routing"]["reason"])
print()
print("Department       :", answers["department"]["choice"],
      f"(confidence: {answers['department']['confidence']:.2f})")
print("Urgency          :", answers["urgency"]["score"])
print("Churn risk       :", f"{answers['churn_risk']['noul']:.3f}")
print("Refund requested :", f"{answers['refund_requested']['noul']:.3f}")

# Multilingual example: same questions, a Hindi ticket body, routed
# automatically to the multilingual checkpoint.
hindi_state = {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}
res_hi = router.predict(hindi_state, questions)
print()
print("Hindi ticket routed to:", res_hi["routing"]["model"])
print("Department             :", res_hi["answers"]["department"]["choice"])
