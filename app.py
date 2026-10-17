import json
import os
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "clean_orders.json"

MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")
TIMEZONE = ZoneInfo("Asia/Kolkata")

RATES = {
    "USD": 1.0,
    "EUR": 1.09,
    "INR": 0.01185,
}

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://aipipe.org/openai/v1",
    ),
)

app = FastAPI(title="Product Ledger API")


# --------------------------------------------------
# 2. Load and clean the ledger once at startup
# --------------------------------------------------

def load_orders():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Accept either a list of orders or an object
    # containing an "orders" list.
    if isinstance(data, dict):
        data = data.get("orders", [])

    if not isinstance(data, list):
        raise ValueError("clean_orders.json must contain a list of orders")

    # Keep the most recently updated version of each order.
    latest = {}

    for order in data:
        order_id = str(order["id"])
        previous = latest.get(order_id)

        if previous is None:
            latest[order_id] = order
        else:
            new_time = str(order.get("updated_at", ""))
            old_time = str(previous.get("updated_at", ""))

            if new_time >= old_time:
                latest[order_id] = order

    return list(latest.values())


ORDERS = load_orders()


# --------------------------------------------------
# 3. Helper functions
# --------------------------------------------------

def order_amount_usd(order):
    """Convert an order's amount into USD."""
    currency = str(order.get("currency", "USD")).upper()

    if currency not in RATES:
        raise ValueError(f"Unknown currency: {currency}")

    amount = order.get("amount")

    if amount is None:
        amount = float(order.get("qty", 0)) * float(
            order.get("unit_price", 0)
        )

    return float(amount) * RATES[currency]


def order_business_date(order):
    """Interpret an order timestamp in Asia/Kolkata."""
    raw = order.get("created_at")

    if not raw:
        return None

    if isinstance(raw, datetime):
        timestamp = raw
    else:
        timestamp = datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")
        )

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=TIMEZONE)

    return timestamp.astimezone(TIMEZONE).date()


def parse_date(value):
    if not value:
        return None
    return date.fromisoformat(value)


# --------------------------------------------------
# 4. Ask AI Pipe to interpret the question
# --------------------------------------------------

SYSTEM_PROMPT = """
You convert questions about an order ledger into JSON instructions.

Return ONLY a JSON object with these fields:

{
  "metric": "revenue",
  "region": null,
  "product": null,
  "customer": null,
  "start_date": null,
  "end_date": null,
  "group_by": null,
  "status_filter": null,
  "sort_by": null,
  "limit": null
}

Allowed metric values:
- revenue: total money from paid orders
- refunds: total money from refunded orders
- order_count: number of orders
- customer_count: number of distinct customers
- quantity: total quantity of items

Allowed group_by values:
- null
- product
- customer
- region
- month

Allowed status_filter values:
- null
- paid
- refunded
- void
- all

Allowed sort_by values:
- null
- revenue
- quantity
- order_count

Interpret dates as business dates in Asia/Kolkata.
Use YYYY-MM-DD for dates.

Interpret phrases such as:
- "March 2026" as 2026-03-01 through 2026-03-31
- "in 2026" as 2026-01-01 through 2026-12-31
- "in the North region" as region="North"
- "for the Blender" as product="Blender"

For revenue, use metric="revenue".
For refunds or refunded money, use metric="refunds".
For "how many orders", use metric="order_count".
For distinct customers, use metric="customer_count".
For items sold or units, use metric="quantity".

If the question asks which product/customer/region had the highest
revenue, set the appropriate group_by and sort_by="revenue".
If it asks for the best-selling product by units, use
group_by="product" and sort_by="quantity".

Use status_filter="paid" for revenue and normally for quantity.
Use status_filter="refunded" for refunds.
For an order count, use status_filter="all" unless the question
specifically asks for paid, refunded, or void orders.

If the question requests a top-N ranking, set limit to N.
If it asks for a single highest or lowest result, set limit to 1.

Do not invent dates, products, customers, or regions.
Use null for filters that the question does not specify.
"""


def interpret_question(question):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )

    instructions = json.loads(
        response.choices[0].message.content
    )

    allowed_metrics = {
        "revenue",
        "refunds",
        "order_count",
        "customer_count",
        "quantity",
    }

    if instructions.get("metric") not in allowed_metrics:
        raise ValueError("AI returned an unsupported metric")

    return instructions


# --------------------------------------------------
# 5. Filter orders and calculate the answer
# --------------------------------------------------

def calculate_answer(instructions):
    metric = instructions["metric"]
    status_filter = instructions.get("status_filter")

    # Apply the appropriate default status.
    if status_filter is None:
        if metric == "revenue":
            status_filter = "paid"
        elif metric == "refunds":
            status_filter = "refunded"
        elif metric == "quantity":
            status_filter = "paid"
        else:
            status_filter = "all"

    start_date = parse_date(instructions.get("start_date"))
    end_date = parse_date(instructions.get("end_date"))

    selected = []

    for order in ORDERS:
        status = str(order.get("status", "")).lower()

        if status_filter != "all" and status != status_filter:
            continue

        region = instructions.get("region")
        if region and str(order.get("region", "")).casefold() != region.casefold():
            continue

        product = instructions.get("product")
        if product and str(order.get("product", "")).casefold() != product.casefold():
            continue

        customer = instructions.get("customer")
        if customer and str(order.get("customer", "")).casefold() != customer.casefold():
            continue

        business_date = order_business_date(order)

        if start_date and (
            business_date is None or business_date < start_date
        ):
            continue

        if end_date and (
            business_date is None or business_date > end_date
        ):
            continue

        selected.append(order)

    def value_for(order):
        if metric in ("revenue", "refunds"):
            return order_amount_usd(order)

        if metric == "quantity":
            return float(order.get("qty", 0))

        return 1

    def aggregate(orders):
        if metric == "customer_count":
            return len({
                str(order.get("customer", ""))
                for order in orders
            })

        if metric == "order_count":
            return len(orders)

        return sum(value_for(order) for order in orders)

    group_by = instructions.get("group_by")

    if group_by:
        groups = defaultdict(list)

        for order in selected:
            if group_by == "month":
                business_date = order_business_date(order)
                if business_date is None:
                    continue
                key = business_date.strftime("%Y-%m")
            else:
                key = str(order.get(group_by, "Unknown"))

            groups[key].append(order)

        results = [
            {
                "name": key,
                "value": aggregate(group_orders),
            }
            for key, group_orders in groups.items()
        ]

        sort_by = instructions.get("sort_by")

        if sort_by:
            def sort_value(item):
                if sort_by == "quantity":
                    return sum(
                        float(order.get("qty", 0))
                        for order in groups[item["name"]]
                    )

                if sort_by == "order_count":
                    return len(groups[item["name"]])

                return sum(
                    order_amount_usd(order)
                    for order in groups[item["name"]]
                    if str(order.get("status", "")).lower() == "paid"
                )

            results.sort(
                key=lambda item: (
                    sort_value(item),
                    item["name"],
                ),
                reverse=True,
            )
        else:
            results.sort(key=lambda item: item["name"])

        limit = instructions.get("limit")

        if isinstance(limit, int) and limit > 0:
            results = results[:limit]

        # Return just the product name when asked for the
        # single top-selling product by revenue.
        if (
            group_by == "product"
            and instructions.get("sort_by") == "revenue"
            and limit == 1
            and results
        ):
            return results[0]["name"]

        if metric in ("revenue", "refunds"):
            results = [
                {"name": item["name"], "value": round(item["value"], 2)}
                for item in results
            ]

        return results

    result = aggregate(selected)

    if metric in ("revenue", "refunds"):
        return round(result, 2)

    if metric in ("order_count", "customer_count"):
        return int(result)

    return result


# --------------------------------------------------
# 6. HTTP API
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Product Ledger API",
        "orders_loaded": len(ORDERS),
    }


@app.post("/")
def answer_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please provide a question.",
        )

    try:
        instructions = interpret_question(question)
        answer = calculate_answer(instructions)

        return {"answer": answer}

    except Exception as error:
        # Log the actual error in the server terminal.
        print(f"Question processing failed: {error}")

        raise HTTPException(
            status_code=500,
            detail="Could not process the question. Check the server logs.",
        )


# --------------------------------------------------
# 7. Allow running this file directly for a quick test
# --------------------------------------------------

if __name__ == "__main__":
    test_question = (
    "In February 2026, what was the top-selling product by revenue (in USD)?"
)

    instructions = interpret_question(test_question)
    answer = calculate_answer(instructions)

    print("Question:", test_question)
    print("Interpreted instructions:", json.dumps(instructions))
    print("Answer:", json.dumps({"answer": answer}))