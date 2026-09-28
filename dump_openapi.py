"""Print the API's OpenAPI document without starting a server.

    venv/Scripts/python.exe dump_openapi.py > openapi.json
    venv/Scripts/python.exe dump_openapi.py --out ../basic-saas-starter/openapi.full.json

The frontend's `npm run gen:api` reads this (or a running backend's
/openapi.json) to regenerate src/types/api.generated.ts. Importing main_api
touches no network: nothing here calls stats.nba.com.
"""
import argparse
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    logging.disable(logging.INFO)   # main_api logs its config at import
    import main_api
    doc = json.dumps(main_api.app.openapi(), indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(doc + "\n")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
