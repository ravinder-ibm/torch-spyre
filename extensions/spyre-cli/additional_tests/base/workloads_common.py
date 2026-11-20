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


"""What the additional spyre-cli suites define: examples, inputs and helpers.

A suite is a folder ``additional_tests/<group>/<suite>/`` holding
``<suite>_examples.py``. That module defines ``EXAMPLES`` (a dict of
:class:`Example`) and the functions the harness calls on it:

    op_fn(name)               the torch callable that is compiled
    make_inputs(name)         CPU inputs, in kernel argument order
    expected(name, inputs)    CPU reference result
    to_device(name, inputs)   optional; device tensors fed to the compiled op
                              and to the launch (default: ``.to("spyre")``)
    check_extra(name, inputs, got)
                              optional; returns an error string or None, for
                              exact checks beyond the tolerance comparison
    METRICS                   optional; {metric name: fn(mod, out, iters)} for the
                              throughput / format checks of the suite
    sdk_launch(name, dev, path)
                              optional; for kernels whose tensors or output layout
                              ``spyre_cli.launch`` cannot take by default. Returns
                              ``(got, runner)``: the result on the host and the
                              runner, which the harness keeps alive until then

Building and checking is done by ``check.py`` and driven, one process per
example, by ``run_all.py``.

    Example                         one registry entry
    suite_names(), load_examples()  find and import a suite
    load_suite()                    import ``<suite>_examples.py``
    torch_dtype(), tensor_spec()    dtype names, ``spyre launch`` tensor arguments
    cli_args(), bundle_arg_count()  the ``spyre launch`` command of an example
    zero_mask(), fp8_*()            building blocks shared by the suites

Importing this module does not import torch, so tests can be collected without
a Spyre device.
"""

import dataclasses
import importlib.util
import regex as re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent  # additional_tests/base
ADDITIONAL = BASE.parent  # additional_tests
DEFAULT_GROUP = "precision-scaling-structured-sparsity-workloads"

# spyre launch tensor types (spyre_cli.core.dtype_mapping).
LAUNCH_DTYPES = ("fp16", "fp32", "bf16")

DEFAULT_TOL = (0.1, 0.1)  # (atol, rtol); the spyre-cli examples' default


@dataclasses.dataclass(frozen=True)
class Example:
    title: str
    workload: str  # spec workload id, e.g. "WK-SPR-02"
    case: str  # test case id in 2.4_missing_testcases.xlsx, e.g. "SPR-02-a"
    inputs: tuple  # input shapes, in kernel argument order
    output: tuple  # output shape
    dtype: str = "fp16"  # dtype of the tensors given to the kernel
    out_dtype: str = ""  # dtype of the output; "" means same as dtype
    tol: tuple = DEFAULT_TOL  # (atol, rtol) against the CPU reference
    xfail: str = ""  # reason this example is expected not to compile

    @property
    def atol(self):
        return self.tol[0]

    @property
    def rtol(self):
        return self.tol[1]

    @property
    def result_dtype(self):
        return self.out_dtype or self.dtype

    @property
    def launchable(self):
        """Whether ``spyre launch`` has a tensor type for every tensor."""
        return self.dtype in LAUNCH_DTYPES and self.result_dtype in LAUNCH_DTYPES


# --------------------------------------------------------------------------
# Finding suites
# --------------------------------------------------------------------------


def suite_names(group=DEFAULT_GROUP):
    """Names of the suites in ``additional_tests/<group>/``, sorted."""
    folder = ADDITIONAL / group
    if not folder.is_dir():
        return []
    return sorted(
        p.name for p in folder.iterdir() if (p / f"{p.name}_examples.py").is_file()
    )


def suite_dir(suite, group=DEFAULT_GROUP):
    return ADDITIONAL / group / suite


def load_suite(suite, group=DEFAULT_GROUP):
    """The module ``<group>/<suite>/<suite>_examples.py``."""
    key = f"{suite}_examples"
    if key not in sys.modules:
        path = suite_dir(suite, group) / f"{key}.py"
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def load_examples(suite, group=DEFAULT_GROUP):
    """The ``EXAMPLES`` dict of a suite."""
    return load_suite(suite, group).EXAMPLES


# --------------------------------------------------------------------------
# dtypes and the spyre launch command
# --------------------------------------------------------------------------


def torch_dtype(dtype):
    import torch

    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }[dtype]


def tensor_spec(shape, dtype):
    """Format a shape as a ``spyre launch`` tensor argument, e.g. 512x1024@bf16."""
    return "x".join(str(d) for d in shape) + "@" + dtype


def cli_args(ex):
    """The ``-i``/``-o`` arguments for launching example ``ex``."""
    if not ex.launchable:
        raise ValueError(f"{ex.title}: spyre launch has no {ex.dtype} tensor type")
    args = []
    for shape in ex.inputs:
        args += ["-i", tensor_spec(shape, ex.dtype)]
    return args + ["-o", tensor_spec(ex.output, ex.result_dtype)]


def bundle_arg_count(kernel_dir):
    """Tensor arguments the compiled kernel takes, or None if not readable.

    ``spyre launch`` passes exactly the ``-i`` and ``-o`` tensors, so a kernel is
    launchable from the CLI only if this equals their number.
    """
    try:
        text = (Path(kernel_dir) / "bundle.mlir").read_text()
    except OSError:
        return None
    sig = re.search(r"func\.func @sdsc_bundle\(([^)]*)\)", text)
    return len(set(re.findall(r"%arg_(\d+)_base_addr", sig.group(1)))) if sig else None


# --------------------------------------------------------------------------
# Building blocks shared by the suites
# --------------------------------------------------------------------------


def zero_mask(shape, fraction, gen):
    """Boolean mask with exactly round(fraction * numel) True entries."""
    import torch

    n = 1
    for d in shape:
        n *= d
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.randperm(n, generator=gen)[: round(fraction * n)]] = True
    return mask.reshape(shape)


def fp8_to_device(inputs, fmt="e4m3"):
    """Device FP8 tensors for the (activation, weight) FP16 CPU ``inputs``.

    E4M3 goes through the spyre quantize ops (scale 1), which produce the
    ``qfp8ch`` / ``qfp8wt`` layouts that ``_scaled_mm`` consumes; they are
    compiled in the same process, before the op under test. E5M2 has no
    quantize op, so it is a plain dtype cast and copy.
    """
    import torch

    a, b = inputs
    if fmt == "e5m2":
        return [t.to(torch.float8_e5m2).to("spyre") for t in (a, b)]
    one = torch.ones(1, dtype=torch.float16, device="spyre")
    quant_a = torch.compile(lambda x, s: torch.ops.spyre.quantize_fp8_with_scale(x, s))
    quant_b = torch.compile(
        lambda x, s: torch.ops.spyre.quantize_weight_fp8_with_scale(x, s)
    )
    return [quant_a(a.to("spyre"), one), quant_b(b.to("spyre"), one)]


def scaled_mm_fn():
    """FP8 [M, K] x [K, N] -> FP16 matmul, as in test_fp8_scaled_mm_*."""
    import torch

    return lambda a, b: torch.ops.aten._scaled_mm(
        a, b, scale_a=None, scale_b=None, bias=None, out_dtype=torch.float16
    )


def fp8_reference(a, b, fmt="e4m3"):
    """fp32 matmul of the FP8-rounded operands, cast to FP16."""
    import torch

    dt = torch.float8_e5m2 if fmt == "e5m2" else torch.float8_e4m3fn
    top = torch.finfo(dt).max
    qa, qb = (t.float().clamp(-top, top).to(dt).float() for t in (a, b))
    return (qa @ qb).to(torch.float16)
