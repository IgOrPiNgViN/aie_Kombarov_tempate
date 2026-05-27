from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    p = argparse.ArgumentParser(description="Client for AIE semantic search service")
    p.add_argument("--base-url", default="http://localhost:8000", help="API base url")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--endpoint",
        choices=["search", "predict"],
        default="search",
        help="Use /search (theme 5.2) or /predict (course checklist alias)",
    )
    args = p.parse_args()

    path = "/search" if args.endpoint == "search" else "/predict"
    r = httpx.post(
        f"{args.base_url}{path}",
        json={"query": args.query, "top_k": args.top_k},
        timeout=60,
    )
    r.raise_for_status()
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
