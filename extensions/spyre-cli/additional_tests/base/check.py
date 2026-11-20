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


"""Build, check and time the examples of the precision / sparsity suites.

These examples cannot go through upstream's ``examples.py`` / ``check.py``: their
operands are not all ``fp16`` / ``fp32`` / ``bf16`` tensors (FP8 operands are made
on the device by the quantize ops, the FP4 and 2:4 formats are packed), their
kernels may take a different number of tensors than ``spyre launch`` passes, and
some need an exact check beyond ``allclose`` (addend pass-through of the zero
gating, the 2:4 pattern of the weights). Three functions, mirroring upstream's
pair and the line ``run_all.py`` parses:

    compile_one(mod, name, out)   torch.compile the op on the device, compare it
                                  with the CPU, copy the one kernel it added to
                                  out/name (nothing is written if it differs)
    check_sdk(mod, name, path)    spyre_cli.launch over the example's inputs and
                                  compare with the CPU; returns (ok, detail)
    bench_one(mod, name, path, iters)
                                  seconds per launch, for the throughput metrics

``mod`` is the suite module (``<suite>_examples.py``).
"""

import glob
import os
import shutil
import sys
import time
from pathlib import Path

from workloads_common import bundle_arg_count, cli_args, torch_dtype


def _kernel_dirs(cache):
    return set(glob.glob(os.path.join(cache, "inductor-spyre", "*", "spyreCodeDir")))


def device_inputs(mod, name, inputs):
    """Device tensors for the CPU ``inputs``: ``mod.to_device`` or ``.to("spyre")``."""
    convert = getattr(mod, "to_device", None)
    if convert is not None:
        return convert(name, inputs)
    return [t.to("spyre") for t in inputs]


def verify(mod, name, inputs, got):
    """Compare ``got`` with the CPU reference.

    Returns ``(problem, max_abs_diff)``; ``problem`` is None when the result
    matches within the example's (atol, rtol) and passes ``mod.check_extra``.
    """
    import torch

    ex = mod.EXAMPLES[name]
    if tuple(got.shape) != tuple(ex.output):
        return f"output shape {tuple(got.shape)} != {ex.output}", float("nan")
    want = mod.expected(name, inputs)
    diff = (got.float() - want.float()).abs()
    max_diff = diff.max().item()
    if not torch.allclose(got.float(), want.float(), atol=ex.atol, rtol=ex.rtol):
        msg = (
            f"result does not match CPU (max abs diff {max_diff:g}, "
            f"{int((diff > ex.atol).sum())} elements differ)"
        )
        return msg, max_diff
    extra = getattr(mod, "check_extra", None)
    if extra is not None:
        problem = extra(name, inputs, got)
        if problem:
            return problem, max_diff
    return None, max_diff


def compile_one(mod, name, out):
    """Compile example ``name`` into ``out/name``. Runs in a child process whose
    TORCHINDUCTOR_CACHE_DIR is already set."""
    import torch

    ex = mod.EXAMPLES[name]
    inputs = mod.make_inputs(name)
    dev = device_inputs(mod, name, inputs)
    cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    # Preparing the inputs may itself compile kernels (FP8 quantization); the
    # kernel of the op under test is the one that appears after this point.
    before = _kernel_dirs(cache)
    got = torch.compile(mod.op_fn(name), backend="inductor")(*dev).cpu()
    problem, _ = verify(mod, name, inputs, got)
    if problem:
        sys.exit(f"{name}: torch.compile {problem}")

    new = _kernel_dirs(cache) - before
    if len(new) != 1:
        sys.exit(f"{name}: expected 1 kernel, torch.compile produced {len(new)}")

    dest = Path(out) / name
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.dirname(next(iter(new))), dest)
    print(f"{name}: wrote {dest}")
    if not ex.launchable:
        print(f"{name}: {ex.dtype} is not a spyre launch tensor type; not launchable")
        return
    wanted = len(ex.inputs) + 1  # spyre launch passes the inputs, then one output
    actual = bundle_arg_count(dest)
    if actual is not None and actual != wanted:
        print(
            f"{name}: spyre launch cannot run this kernel: it takes {actual} "
            f"tensors and the CLI passes {wanted}; the SDK check runs it"
        )
        return
    print(f"{name}: spyre launch {' '.join(cli_args(ex))} {dest}")


def check_sdk(mod, name, path):
    """Run the compiled kernel through the spyre_cli SDK and compare with CPU.

    By default the kernel gets the inputs then one output allocated with
    ``torch.empty``. A suite whose kernel takes other tensors or a different
    output layout defines ``sdk_launch(name, dev, path)``, which returns
    ``(got, runner)``: the result read back to the host and the runner to keep
    alive until then.
    """
    import spyre_cli
    import torch

    ex = mod.EXAMPLES[name]
    inputs = mod.make_inputs(name)
    dev = device_inputs(mod, name, inputs)
    launch = getattr(mod, "sdk_launch", None)
    try:
        if launch is not None:
            got, runner = launch(name, dev, path)
        else:
            out = torch.empty(
                ex.output, dtype=torch_dtype(ex.result_dtype), device="spyre"
            )
            runner = spyre_cli.launch(*dev, out, path=path)
            got = out.cpu()
        del runner  # only after the outputs are read back
    except Exception as exc:  # noqa: BLE001  report it; the table would show "no result"
        return False, f"launch raised {type(exc).__name__}: {str(exc)[:500]}"
    problem, max_diff = verify(mod, name, inputs, got)
    if problem:
        return False, problem
    return True, f"matches CPU, max abs diff {max_diff:g}"


def bench_one(mod, name, kernel_dir, iters):
    """Seconds per launch of the compiled kernel, over ``iters`` launches.

    One runner is reused and the output is read back once at the end to wait for
    the queue to drain, so the figure includes one device-to-host copy of the
    output (identical across examples of the same output shape).
    """
    import torch
    from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

    ex = mod.EXAMPLES[name]
    dev = device_inputs(mod, name, mod.make_inputs(name))
    out = torch.empty(ex.output, dtype=torch_dtype(ex.result_dtype), device="spyre")
    runner = SpyreSDSCKernelRunner("spyre-cli", str(kernel_dir))
    runner.run(*dev, out)  # warm-up
    out.cpu()
    start = time.perf_counter()
    for _ in range(iters):
        runner.run(*dev, out)
    out.cpu()
    elapsed = time.perf_counter() - start
    del runner
    return elapsed / iters
