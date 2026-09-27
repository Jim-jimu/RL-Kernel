"""Offline CLI and optional GPU reference validation, with explicit evidence scope."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest

from .contract import ContractError, POLICY, SCHEMA, digest, exact_keys, require
from .fixtures import Generator, evaluate, load_golden, make_case, make_golden
from .oracle import forward, mock_return
from .contract import Plan

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "fixtures" / "golden.json"


def source_hash():
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(ROOT.rglob("*.py"))}
    return digest(files)


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def verify_record(record):
    require(evaluate(record["input"]) == record["expected"], "BYTE_MISMATCH", record["input"]["name"])
    case = record["input"]
    plan = Plan.from_dict(case["plan"])
    for arrival in (list(reversed(range(len(case["rows"])))), Generator(19).shuffle(range(len(case["rows"])))):
        returned = mock_return(plan, case["rows"], arrival, plan.context)
        fwd = forward(plan, returned, case["shared"], case["residual"], plan.context)
        require(fwd["stages"] == record["expected"]["forward"], "BYTE_MISMATCH", "arrival changed arithmetic")
    return {"case": case["name"], "verdict": "SCALAR_ORACLE_PASS", "arrival_schedules": 3}


def artifact_write(output, recordings, report):
    output = Path(output)
    # An attempt is immutable; no silent overwrite/resume.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "recordings.json", recordings)
    write_json(output / "report.json", report)
    payload = {"schema": SCHEMA, "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                          for name in ("recordings.json", "report.json")}}
    write_json(output / "completion.json", {"payload": payload, "seal_sha256": digest(payload)})


def artifact_verify(output):
    output = Path(output)
    seal = json.loads((output / "completion.json").read_text())
    exact_keys(seal, ("payload", "seal_sha256"), "completion envelope")
    require(digest(seal["payload"]) == seal["seal_sha256"], "CORRUPT_ARTIFACT", "seal digest")
    require(seal["payload"]["schema"] == SCHEMA, "SCHEMA_MISMATCH", "completion schema")
    exact_keys(seal["payload"]["files"], ("recordings.json", "report.json"), "sealed filenames")
    for name, sha in seal["payload"]["files"].items():
        require(hashlib.sha256((output / name).read_bytes()).hexdigest() == sha,
                "CORRUPT_ARTIFACT", name)
    recorded = json.loads((output / "recordings.json").read_text())
    require(recorded["schema"] == SCHEMA and recorded["policy"] == POLICY,
            "SCHEMA_MISMATCH", "recording profile")
    for record in recorded["cases"]:
        verify_record(record)
    report = json.loads((output / "report.json").read_text())
    require(report["production_certified"] is False and report["foundation_compatibility"] == "UNVERIFIED",
            "SCHEMA_MISMATCH", "draft cannot certify production")
    require(report["recordings_sha256"] == digest(recorded), "CORRUPT_ARTIFACT", "report binding")
    return {"status": "ARTIFACT_INTEGRITY_AND_CPU_REPLAY_PASS", "cases": len(recorded["cases"]),
            "original_scope": report["scope"], "gpu_reexecuted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    sub.add_parser("contract")
    freeze = sub.add_parser("freeze-goldens", help="Maintainer only; refuses to overwrite")
    freeze.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("directory")
    conf = sub.add_parser("conformance")
    conf.add_argument("--device", default="cpu", help="cpu=stdlib oracle; torch-cpu; cuda:0")
    conf.add_argument("--require-h100", action="store_true")
    conf.add_argument("--graph", action="store_true")
    conf.add_argument("--hidden-size", type=int, default=0, help="extra synthetic tail case; 4096 on H100")
    conf.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests")))
            print(json.dumps({"status": "PASS" if result.wasSuccessful() else "FAIL",
                              "tests": result.testsRun, "failures": len(result.failures),
                              "errors": len(result.errors), "skipped": len(result.skipped),
                              "scope": "CPU_UNIT_ONLY", "production_certified": False}))
            return 0 if result.wasSuccessful() else 1
        if args.command == "contract":
            print(json.dumps({"schema": SCHEMA, "policy": POLICY}, indent=2))
            return 0
        if args.command == "freeze-goldens":
            write_json(args.output, make_golden())
            print(json.dumps({"status": "GOLDENS_WRITTEN", "output": args.output}))
            return 0
        if args.command == "verify":
            print(json.dumps(artifact_verify(args.directory), indent=2))
            return 0
        require(not Path(args.output).exists(), "OUTPUT_EXISTS", "use a fresh attempt directory")
        require(args.hidden_size >= 0, "UNSUPPORTED_GEOMETRY", "hidden-size")
        golden = load_golden(GOLDEN)
        recorded = golden["payload"]
        if args.hidden_size:
            case = make_case(f"extra-width-{args.hidden_size}", n=2, h=args.hidden_size, mode="mixed", seed=31)
            recorded["cases"].append({"input": case, "expected": evaluate(case)})
        results = [verify_record(r) for r in recorded["cases"]]
        provenance = {"provider": "stdlib-scalar-oracle", "python": platform.python_version(), "device": "cpu"}
        if args.device == "cpu":
            require(not args.require_h100 and not args.graph, "UNSUPPORTED_CAPABILITY", "CPU cannot satisfy GPU request")
        else:
            from .torch_reference import conformance
            device = "cpu" if args.device == "torch-cpu" else args.device
            provenance, results = conformance(recorded["cases"], device, args.require_h100, args.graph)
        report = {"status": "REFERENCE_CONFORMANCE_PASS", "scope": "T01_REFERENCE_ONLY",
                  "schema": SCHEMA, "source_sha256": source_hash(),
                  "recordings_sha256": digest(recorded), "golden_sha256": golden["payload_sha256"],
                  "foundation_compatibility": "UNVERIFIED", "production_certified": False,
                  "hardware_execution": args.device.startswith("cuda"),
                  "provenance": provenance, "cases": results, "performance": "NOT_MEASURED"}
        artifact_write(args.output, recorded, report)
        print(json.dumps({"status": report["status"], "output": args.output,
                          "checks": len(results), "scope": report["scope"], "provenance": provenance}, indent=2))
        return 0
    except (ContractError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "code": getattr(exc, "code", type(exc).__name__),
                          "detail": str(exc), "production_certified": False}), file=sys.stderr)
        return 1
