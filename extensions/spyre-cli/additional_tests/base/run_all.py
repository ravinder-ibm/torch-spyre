# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Run the additional spyre-cli suites one by one, through the SDK.

Usage: python run_all.py [--group G] [--out DIR] [--list] [--metrics]
                         [--iters N] [suite | example ...]

With no arguments every suite of the group (zero_gating, fp8_gemm, ...) is run,
in order. Name suites to run only those, or examples (names are unique across
suites) to run only those.

For each example this script runs, in a fresh process:

1. build: ``check.py``'s ``compile_one``  (torch.compile on the device, compared
   with the CPU, the one kernel copied to <DIR>/<suite>/<example>; default
   <suite>/build/<example>)
2. check: ``check.py``'s ``check_sdk``  (spyre_cli.launch on the example's inputs,
   must match the CPU reference within the example's (atol, rtol) and pass the
   suite's exact check, if it has one)

A fresh process per example matters: a device fault leaves the stream in an
error state for the rest of the process it happens in. A failing example does not
stop the run; if its build fails there is no bundle, so its check is reported as
SKIP. An example with an ``xfail`` reason is expected not to build: that is
XFAIL and counts as passing; if it builds after all it is XPASS (its ``xfail``
can be dropped) and is checked like any other. A summary table is printed at the
end, and the exit status is 0 only if every selected example is as expected.

``--metrics`` runs the throughput and format metrics a suite defines in its
``METRICS`` (TOPS ratio, sparse speedup, weight compression) on the kernels
already built in the output directory, in a fresh process each.
"""

import argparse
import os
import regex as re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workloads_common import (
    DEFAULT_GROUP,
    load_examples,
    load_suite,
    suite_dir,
    suite_names,
)

SELF = Path(__file__).resolve()
OK_SDK = ("PASS", "SKIP")  # SKIP: an xfail example that did not build
RESULT_LINE = re.compile(r"^(\S+) sdk: (PASS|FAIL) ?(.*)$", re.MULTILINE)


# --------------------------------------------------------------------------
# Child processes: one build, one check or one metric
# --------------------------------------------------------------------------


def _child(mode, group, suite, target, out, iters="20"):
    mod = load_suite(suite, group)
    if mode == "metric":
        ok, detail = mod.METRICS[target](mod, Path(out), int(iters))
        print(f"{target} metric: {'PASS' if ok else 'FAIL'} {detail}")
        sys.exit(0 if ok else 1)

    import check

    if mode == "build":
        check.compile_one(mod, target, out)  # exits non-zero on failure
        return
    ok, detail = check.check_sdk(mod, target, Path(out) / target)
    print(f"{target} sdk: {'PASS' if ok else 'FAIL'} {detail}")
    sys.exit(0 if ok else 1)


# --------------------------------------------------------------------------
# Parent: run every selected example, then summarise
# --------------------------------------------------------------------------


def _run(cmd, timeout, env=None):
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False
    )
    return result.returncode, result.stdout + result.stderr


def _status(output):
    """PASS/FAIL (and detail) of the ``<name> sdk:`` line in check output."""
    match = RESULT_LINE.search(output)
    return (match.group(2), match.group(3)) if match else ("FAIL", "no result")


def _reason(name, output):
    """The most informative line of a failed build's output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    own = [line for line in lines if name in line and not line.startswith("failed:")]
    errors = [line for line in lines if re.match(r"\w*(Error|Exception)\b", line)]
    return (own or errors or lines or ["compile failed"])[-1]


def run_one(group, suite, name, out, xfail=""):
    """Build and check ``name``; return its result row."""
    row = {"compile": "FAIL", "sdk": "SKIP", "detail": ""}
    with tempfile.TemporaryDirectory() as cache:
        # A private Inductor cache, so exactly one kernel directory is produced.
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=cache)
        cmd = [sys.executable, str(SELF), "--child", "build", group, suite, name, out]
        try:
            code, output = _run(cmd, 1800, env)
        except subprocess.TimeoutExpired:
            row["detail"] = "compile timed out"
            return row
    if code != 0:
        row["compile"] = "XFAIL" if xfail else "FAIL"
        row["detail"] = xfail or _reason(name, output)
        return row
    row["compile"] = "XPASS" if xfail else "PASS"

    cmd = [sys.executable, str(SELF), "--child", "check", group, suite, name, out]
    try:
        _, output = _run(cmd, 1200)
    except subprocess.TimeoutExpired:
        row["sdk"] = "FAIL"
        row["detail"] = "check timed out"
        return row
    row["sdk"], row["detail"] = _status(output)
    if xfail:
        row["detail"] = f"expected to fail ({xfail}) but compiled; {row['detail']}"
    return row


def passed(row):
    return row["compile"] in ("PASS", "XPASS", "XFAIL") and row["sdk"] in OK_SDK


def select(group, targets):
    """``[(suite, example name)]`` for the suite and example names in ``targets``."""
    suites = {s: load_examples(s, group) for s in suite_names(group)}
    if not targets:
        targets = list(suites)
    chosen, unknown = [], []
    for target in targets:
        if target in suites:
            chosen += [(target, n) for n in suites[target]]
            continue
        owners = [s for s, examples in suites.items() if target in examples]
        if owners:
            chosen.append((owners[0], target))
        else:
            unknown.append(target)
    return suites, chosen, unknown


def out_dir(args, suite):
    return str(
        Path(args.out) / suite if args.out else suite_dir(suite, args.group) / "build"
    )


def run_metrics(args, suites):
    """Run every metric of the selected suites; return the failed ones."""
    failed = []
    for suite in suites:
        for metric in getattr(load_suite(suite, args.group), "METRICS", {}):
            print(f"== {suite}/{metric}", flush=True)
            cmd = [
                sys.executable, str(SELF), "--child", "metric", args.group, suite,
                metric, out_dir(args, suite), str(args.iters),
            ]  # fmt: skip
            try:
                _, output = _run(cmd, 1800)
            except subprocess.TimeoutExpired:
                output = f"{metric} metric: FAIL timed out"
            match = re.search(
                rf"^{metric} metric: (PASS|FAIL) ?(.*)$", output, re.MULTILINE
            )
            status, detail = match.groups() if match else ("FAIL", _reason("", output))
            print(f"   {status}  {detail}", flush=True)
            if status != "PASS":
                failed.append(f"{suite}/{metric}")
    return failed


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(*sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", metavar="suite|example")
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--out", help="kernel directory (default: <suite>/build)")
    parser.add_argument("--list", action="store_true", help="list the examples")
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="run the suites' throughput / format metrics instead of the examples",
    )
    parser.add_argument("--iters", type=int, default=20, help="launches per timing")
    args = parser.parse_args()

    suites, chosen, unknown = select(args.group, args.targets)
    if unknown:
        parser.error(
            f"unknown suite or example: {' '.join(unknown)} "
            f"(suites: {' '.join(suites)})"
        )
    if not chosen:
        parser.error(f"no suites found in group {args.group!r}")

    if args.list:
        for suite, name in chosen:
            ex = suites[suite][name]
            flag = " [xfail]" if ex.xfail else ""
            print(f"{name:24s} {suite:12s} {ex.workload} {ex.case:9s} {ex.title}{flag}")
        return

    if args.metrics:
        failed = run_metrics(args, dict.fromkeys(s for s, _ in chosen))
        if failed:
            sys.exit("failed: " + " ".join(failed))
        return

    rows = {}
    for suite, name in chosen:
        ex = suites[suite][name]
        print(f"== {suite}/{name}", flush=True)
        rows[name] = run_one(args.group, suite, name, out_dir(args, suite), ex.xfail)
        print(f"   {'PASS' if passed(rows[name]) else 'FAIL'}", flush=True)

    print()
    header = (
        f"{'Suite':<12} {'Workload':<10} {'Case':<9} {'Example':<20} "
        f"{'compile':<8} {'sdk':<5} detail"
    )
    print(header)
    print("-" * len(header))
    for suite, name in chosen:
        ex, row = suites[suite][name], rows[name]
        print(
            f"{suite:<12} {ex.workload:<10} {ex.case:<9} {name:<20} "
            f"{row['compile']:<8} {row['sdk']:<5} {row['detail']}"
        )
    ok = sum(passed(row) for row in rows.values())
    print(f"\n{ok}/{len(chosen)} examples as expected")
    xpass = [n for _, n in chosen if rows[n]["compile"] == "XPASS"]
    if xpass:
        print("XPASS (drop the xfail): " + " ".join(xpass))
    failed = [n for _, n in chosen if not passed(rows[n])]
    if failed:
        print("failed: " + " ".join(failed))
    sys.exit(0 if ok == len(chosen) else 1)


if __name__ == "__main__":
    main()
