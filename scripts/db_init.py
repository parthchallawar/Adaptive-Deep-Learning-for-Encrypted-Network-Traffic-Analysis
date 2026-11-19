"""Create (or check) the service database (spec 018; plan phase 6, T1).

    python scripts/db_init.py                                   # the URL in configs/service.yaml
    python scripts/db_init.py --url sqlite:///results/service/demo.db

Creates every table and records the schema version; on an existing database it checks the version
and raises if another one wrote it. The service does the same at start-up, so this is only needed
to inspect or prepare a file in advance. Prints the row count of every table.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="SQLAlchemy URL (default: service config)")
    parser.add_argument("--config", default="configs/service.yaml")
    args = parser.parse_args(argv)

    from adl_etc.service.db import models as m
    from adl_etc.service.db.store import Store

    url = args.url
    if url is None:
        from adl_etc.utils.config import load_config

        url = str(load_config(args.config).db_url)
    store = Store(url).start()
    try:
        print(f"{url}: schema version {m.SCHEMA_VERSION}")
        for name in ("schema_version", *m.TABLE_ORDER):
            print(f"  {name:<18} {store.count(name):>10,} rows")
    finally:
        store.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
