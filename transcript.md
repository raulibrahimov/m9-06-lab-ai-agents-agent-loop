# Transcript — Lab | Your First Agent

Real output of `python agent_lab.py` (model: `gemini-3-flash-preview` via the
`GEMINI_MODEL` env var — the `2.5-flash` free-tier daily quota was exhausted
at run time). One ADK agent (`orders_assistant`) with instructions, two tools
(`lookup_order`, `calculate`) and an `InMemorySessionService` session; the
agent ran the reason → act → observe loop entirely on its own — no loop code
of mine.

## Multi-step goal

```
======================================================================
GOAL: I'm thinking of buying two more of order A1001. What would those two cost, and is the original still under warranty?
======================================================================
  [orders_assistant] tool call   -> lookup_order({"order_id": "A1001"})
  [orders_assistant] tool result <- {"order_id": "A1001", "item": "laptop", "price": 1200, "purchased": "2026-05-20", "warranty_months": 12}
  [orders_assistant] tool call   -> calculate({"expression": "1200 * 2"})
  [orders_assistant] tool result <- {"expression": "1200 * 2", "result": 2400}
  [orders_assistant] FINAL ANSWER: Two more of the laptop from order A1001 would cost $2,400. The original item is still under warranty, as it expires on May 20, 2027.
```

The trace shows the agent's *own* choices: it looked up the order first,
sent the exact math to `calculate` (per its instructions), and reasoned the
warranty out itself — purchased 2026-05-20 + 12 months = 2027-05-20, which is
after today (2026-07-05), so still active. Nothing in my code decides the
order of those calls.

## Stretch: a goal the agent can't complete

```
======================================================================
GOAL: What did I buy in order A9999, and is it still under warranty?
======================================================================
  [orders_assistant] tool call   -> lookup_order({"order_id": "A9999"})
  [orders_assistant] tool result <- {"error": "order not found: 'A9999'"}
  [orders_assistant] FINAL ANSWER: I'm sorry, but order A9999 could not be found.
```

The agent used the tool, observed the "not found" error, and reported
honestly instead of inventing details — exactly what its instructions
prepared it for.
