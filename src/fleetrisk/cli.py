import argparse
import sys
from typing import Any

from fleetrisk.common.io import dumps
from fleetrisk.config import Settings, contained_path
from fleetrisk.data.contracts import CONTRACTS
from fleetrisk.data.db import make_engine, one, rows
from fleetrisk.data.fixtures import write_ltfs_fixture
from fleetrisk.data.ingestion import ingest
from fleetrisk.data.validation import validate
from fleetrisk.modeling.training import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="fleetrisk")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "ingest"):
        item = sub.add_parser(command)
        item.add_argument("dataset", choices=list(CONTRACTS))
        item.add_argument("relative_path")
        item.add_argument("--provenance", choices=["real", "synthetic"], required=True)
    sub.add_parser("discover")
    for command in ("inspect", "quality", "replay"):
        sub.add_parser(command).add_argument("ingestion_id")
    trainer = sub.add_parser("train")
    trainer.add_argument("snapshot_id")
    trainer.add_argument("--quick", action="store_true")
    sub.add_parser("fixture")
    sub.add_parser("recover-interrupted")
    args = parser.parse_args()
    settings = Settings()
    output: Any
    if args.command == "discover":
        output = sorted(
            str(path.relative_to(settings.source_root))
            for path in settings.source_root.rglob("*")
            if path.suffix.lower() in (".csv", ".parquet") and path.is_file()
        )
    elif args.command == "fixture":
        if not settings.allow_synthetic:
            parser.error("Set FLEETRISK_ALLOW_SYNTHETIC=true for fixture mode")
        output = {
            "path": str(write_ltfs_fixture(settings.source_root / "fixtures" / "ltfs.csv")),
            "provenance": "synthetic",
        }
    elif args.command == "validate":
        output = validate(contained_path(settings.source_root, args.relative_path), args.dataset)
        output["provenance"] = args.provenance
        print(dumps(output))
        if not output["valid"]:
            sys.exit(2)
        return
    else:
        engine = make_engine(settings)
        try:
            if args.command == "ingest":
                output = ingest(engine, settings, args.dataset, args.relative_path, args.provenance)
            elif args.command == "train":
                output = train(engine, settings, args.snapshot_id, quick=args.quick)
            elif args.command == "recover-interrupted":
                # Explicit operator action only: do not mark live workers abandoned automatically.
                with engine.begin() as connection:
                    output = rows(
                        connection,
                        """UPDATE ingestions SET status='failed',error_code='operator_marked_interrupted',finished_at=now()
                        WHERE status='running' AND created_at < now() - interval '24 hours' RETURNING id""",
                    )
            else:
                with engine.connect() as connection:
                    record = one(
                        connection, "SELECT * FROM ingestions WHERE id=:id", id=args.ingestion_id
                    )
                if record is None:
                    raise ValueError("Unknown ingestion")
                if args.command == "replay":
                    output = ingest(
                        engine,
                        settings,
                        record["source_id"],
                        record["relative_path"],
                        record["provenance"],
                        replay_of=args.ingestion_id,
                    )
                else:
                    output = record["quality"] if args.command == "quality" else record
        finally:
            engine.dispose()
    print(dumps(output))


if __name__ == "__main__":
    main()
