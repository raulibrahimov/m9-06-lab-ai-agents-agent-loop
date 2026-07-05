"""Lab | Your First Agent — a single Google ADK agent runs the loop itself.

Yesterday the loop was mine; today the agent owns it. I only *design* the
agent: instructions, two tools, a session — then hand it a multi-step goal
and print the trace (reasoning, tool calls, tool results) as it works.

Usage:
    export GOOGLE_API_KEY="..."   # never committed
    python agent_lab.py
"""

import ast
import asyncio
import datetime
import json
import operator
import os
import re
import sys
import time

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
APP_NAME = "orders_agent_lab"
USER_ID = "raul"
ORDERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.json")

with open(ORDERS_PATH, encoding="utf-8") as f:
    ORDERS = json.load(f)


# --------------------------------------------------------------------------
# Tools — plain Python functions. ADK reads the type hints and docstrings
# to build the schemas the model sees; I don't hand-write them anymore.
# --------------------------------------------------------------------------

def lookup_order(order_id: str) -> dict:
    """Look up an order by its id (e.g. 'A1001').

    Args:
        order_id: The order id to look up, e.g. 'A1001'.

    Returns:
        A dict with item, price (USD), purchased date (YYYY-MM-DD) and
        warranty_months, or a dict with an 'error' key if the order
        does not exist.
    """
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"order not found: {order_id!r}"}
    return {"order_id": order_id, **order}


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> dict:
    """Evaluate a simple arithmetic expression exactly.

    Args:
        expression: An arithmetic expression using numbers and
            + - * / // % ** and parentheses, e.g. '1200 * 2'.

    Returns:
        A dict with the expression and its numeric result, or a dict with
        an 'error' key if the expression is invalid.
    """
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return {"error": f"could not evaluate {expression!r}: {exc}"}
    return {"expression": expression, "result": result}


# --------------------------------------------------------------------------
# The agent — instructions + tools. No loop code of mine anywhere below;
# ADK's Runner lets the agent reason -> act -> observe on its own.
# --------------------------------------------------------------------------

TODAY = datetime.date.today().isoformat()

orders_agent = Agent(
    name="orders_assistant",
    model=MODEL,
    description="A helpful assistant for questions about customer orders.",
    instruction=(
        "You are a helpful orders assistant. Today's date is "
        f"{TODAY}.\n"
        "- Use the lookup_order tool whenever the user mentions an order id; "
        "never guess order details from memory.\n"
        "- Use the calculate tool for ALL arithmetic (totals, multiples, "
        "date math on months you've converted to numbers); do not do exact "
        "math in your head.\n"
        "- For warranty questions: the warranty runs warranty_months from "
        "the purchased date; compare against today's date and say clearly "
        "whether it is still active.\n"
        "- If an order cannot be found, say so clearly and do NOT invent "
        "any details about it.\n"
        "- Keep answers short and concrete."
    ),
    tools=[lookup_order, calculate],
)


def describe_event(event) -> None:
    """Print one trace line per interesting thing in an ADK event."""
    if not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if part.function_call:
            print(f"  [{event.author}] tool call   -> "
                  f"{part.function_call.name}({json.dumps(dict(part.function_call.args))})")
        elif part.function_response:
            print(f"  [{event.author}] tool result <- "
                  f"{json.dumps(dict(part.function_response.response))}")
        elif part.text and part.text.strip():
            label = "FINAL ANSWER" if event.is_final_response() else "thinking"
            print(f"  [{event.author}] {label}: {part.text.strip()}")


async def run_goal(runner: Runner, session_id: str, goal: str,
                   attempts: int = 3) -> None:
    print(f"\n{'=' * 70}\nGOAL: {goal}\n{'=' * 70}")
    message = types.Content(role="user", parts=[types.Part(text=goal)])
    for attempt in range(attempts):
        try:
            async for event in runner.run_async(
                user_id=USER_ID, session_id=session_id, new_message=message
            ):
                describe_event(event)
            return
        except Exception as exc:  # free-tier 429s surface mid-stream
            if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == attempts - 1:
                raise
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
            delay = float(match.group(1)) + 2 if match else 35.0
            print(f"  .. rate limited, retrying whole goal in {delay:.0f}s")
            time.sleep(delay)


async def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Set GOOGLE_API_KEY first.")

    session_service = InMemorySessionService()
    runner = Runner(agent=orders_agent, app_name=APP_NAME,
                    session_service=session_service)

    # A session to hold the multi-step run.
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="run-1"
    )
    await run_goal(
        runner, session.id,
        "I'm thinking of buying two more of order A1001. What would those "
        "two cost, and is the original still under warranty?",
    )

    # Optional stretch: a goal the agent CANNOT complete — it should use the
    # tool, get "not found", and report honestly instead of inventing.
    session2 = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="run-2"
    )
    await run_goal(
        runner, session2.id,
        "What did I buy in order A9999, and is it still under warranty?",
    )


if __name__ == "__main__":
    asyncio.run(main())
