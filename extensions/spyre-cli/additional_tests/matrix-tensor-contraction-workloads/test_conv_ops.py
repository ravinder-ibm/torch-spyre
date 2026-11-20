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


"""WK-MAT-05: Vision 2D Spatial Convolution, as spyre-cli examples.

Workload: B=1, C_in=64, C_out=128, H=W=56, K=3 (a 3x3 conv sliding 128 filters
of shape [64, 3, 3] across a [56, 56] feature map) -- the entry point of a
vision encoder. Precisions: DL16, BF16 and INT8. Quality metrics: stride
boundary integrity and zero border artifacts. Each kernel takes two inputs, the
activation [1, 64, 56, 56] and the weight [128, 64, 3, 3], and has no bias.

    ====================  =========  ======  ==========================
    example               precision  stride  checks
    ====================  =========  ======  ==========================
    conv_dl16_test        DL16       1       random data vs CPU
    conv_test             BF16       1       random data vs CPU
    conv_int8_test        INT8       1       random data vs CPU (exact), xfail
    conv_stride2_test     BF16       2       stride boundary integrity
    conv_border_dl16      DL16       1       zero border artifacts
    conv_border_bf16      BF16       1       zero border artifacts
    conv_border_int8      INT8       1       zero border artifacts, xfail
    ====================  =========  ======  ==========================

* Stride boundary integrity: a bug in the DMA stride logic corrupts the feature
  map at row/column boundaries while the model keeps running. stride=2 puts the
  last window on the boundary (56 + 2 - 3 = 55 is odd), where such a bug shows.
* Zero border artifacts: the ``conv_border_*`` examples feed all-ones
  activations, so the expected output is known in closed form with no conv at
  all: every element is the number of in-bounds taps times the number of
  active input channels. With padding=1 and 64 active channels that is 9*64 in
  the interior, 6*64 on the edges and 4*64 in the corners, so a border
  row/column that reads garbage instead of zero padding is an exact mismatch.
  INT8 activates a single input channel (576 would overflow int8), giving 9, 6
  and 4.

DL16 is the device's native 16-bit format and is what ``torch.float16`` maps to
on Spyre, so on the ``spyre launch`` command line it is ``fp16``. ``spyre
launch`` has no int8 tensor type, so the INT8 examples cannot be launched from
the CLI even when they compile.

The INT8 examples are expected to fail (``xfail``): with padding=1 the conv goes
through the im2col + matmul decomposition (the direct lowering needs padding=0
and fp16), and for int8 the restickify pass raises "no mechanism to resolve
stick incompatibility" on the 64*3 = 192 wide im2col buffer (192 is 3 sticks at
fp16 but 1.5 sticks at int8's 128). An xfail is reported but does not fail the
run; an unexpected pass is reported as XPASS.

Importing this module does not import torch, so it can be collected without a
Spyre device.

Run as a script to compile example kernels on the current toolchain:

    python test_conv_ops.py [--out DIR] [name ...]    (default: every example)

Each example is compiled in a fresh process with its own Inductor cache, so
exactly one Spyre kernel directory is produced, and it is copied to DIR/<name>
(default build/<name>). Examples that compile to more than one kernel are
rejected, since spyre launch runs a single kernel.

``spyre launch`` cannot run the conv kernels. It passes exactly the ``-i`` and
``-o`` tensors, but the compiled conv takes four, in this order (observed from
the generated wrapper of conv_test, ``sdsc_fused_convolution_0.run``):

    activation [1, 64, 56, 56], a scalar zero (the padding fill, from
    ``spyre_constant_tensor``), weight [128, 64, 3, 3], and the output in
    channel-last layout [1, H_out, W_out, 128].

``--launch`` runs a compiled kernel through the spyre_cli SDK with exactly
those tensors, on the same inputs as the compile step, and compares the result
with the CPU reference:

    python test_conv_ops.py conv_test
    python test_conv_ops.py --launch conv_test

    python test_conv_ops.py                       # compile all examples
    for n in conv_dl16_test conv_stride2_test conv_border_dl16 conv_border_bf16; do
      python test_conv_ops.py --launch $n
    done
"""

import argparse
import dataclasses
import glob
import os
import regex as re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "build"

B, C_IN, C_OUT, H, W, K = 1, 64, 128, 56, 56, 3

# (atol, rtol) for random float data. The reference accumulates the
# 64*3*3 = 576 term reduction in fp32 from the already-rounded inputs, so the
# tolerance only has to absorb the 16-bit rounding of the device output.
FLOAT_TOL = (0.25, 0.1)
# INT8 inputs are {-1, 0, 1}: a 576-term sum has a standard deviation of ~16,
# far inside int8's range, so the output cannot overflow and is compared
# exactly. The ones-input border check is exact for every precision.
EXACT_TOL = (0.0, 0.0)

# spyre launch dtype names. "int8" is a workload precision but is not a tensor
# type spyre launch accepts.
_LAUNCH_DTYPES = ("fp16", "fp32", "bf16")

_INT8_XFAIL = (
    "int8 conv does not compile: restickify raises 'no mechanism to resolve "
    "stick incompatibility' (the direct conv lowering is fp16-only)"
)


def _out_size(size, stride, padding):
    return (size + 2 * padding - K) // stride + 1


@dataclasses.dataclass(frozen=True)
class Example:
    title: str
    case: str  # test case id, shown in run_all.py's table
    dtype: str  # "fp16" (DL16), "bf16" or "int8"
    stride: int = 1
    padding: int = 1
    # How inputs are drawn (see make_inputs):
    #   "randn"  random activation and weight, compared with a CPU conv
    #   "ones"   all-ones activation; expected output is analytic (borders)
    init: str = "randn"
    workload: str = "WK-MAT-05"
    xfail: str = ""  # reason this example is expected not to compile

    @property
    def tol(self):
        if self.dtype == "int8" or self.init == "ones":
            return EXACT_TOL
        return FLOAT_TOL

    @property
    def atol(self):
        return self.tol[0]

    @property
    def rtol(self):
        return self.tol[1]

    @property
    def launchable(self):
        """Whether ``spyre launch`` has a tensor type for this precision."""
        return self.dtype in _LAUNCH_DTYPES

    @property
    def inputs(self):
        """Input shapes in kernel argument order: activation, weight."""
        return ((B, C_IN, H, W), (C_OUT, C_IN, K, K))

    @property
    def output(self):
        return (
            B,
            C_OUT,
            _out_size(H, self.stride, self.padding),
            _out_size(W, self.stride, self.padding),
        )


EXAMPLES = {
    # Representative workload in each precision, random data vs CPU.
    "conv_dl16_test": Example("Conv2d 64->128, 3x3 (DL16)", "CONV-01", "fp16"),
    "conv_test": Example("Conv2d 64->128, 3x3 (BF16)", "CONV-02", "bf16"),
    "conv_int8_test": Example(
        "Conv2d 64->128, 3x3 (INT8)", "CONV-03", "int8", xfail=_INT8_XFAIL
    ),
    # Stride boundary integrity.
    "conv_stride2_test": Example(
        "Conv2d 64->128, 3x3, stride 2", "CONV-04", "bf16", stride=2
    ),
    # Zero border artifacts: all-ones input, closed-form expected output.
    "conv_border_dl16": Example(
        "Zero border artifacts (DL16)", "CONV-05", "fp16", init="ones"
    ),
    "conv_border_bf16": Example(
        "Zero border artifacts (BF16)", "CONV-06", "bf16", init="ones"
    ),
    "conv_border_int8": Example(
        "Zero border artifacts (INT8)",
        "CONV-07",
        "int8",
        init="ones",
        xfail=_INT8_XFAIL,
    ),
}


def tensor_spec(shape, dtype):
    """Format a shape as a ``spyre launch`` tensor argument, e.g. 512x1024@bf16."""
    return "x".join(str(d) for d in shape) + "@" + dtype


def cli_args(name):
    """The ``-i``/``-o`` arguments for launching example ``name``."""
    ex = EXAMPLES[name]
    if not ex.launchable:
        raise ValueError(f"{name}: spyre launch has no {ex.dtype} tensor type")
    args = []
    for shape in ex.inputs:
        args += ["-i", tensor_spec(shape, ex.dtype)]
    return args + ["-o", tensor_spec(ex.output, ex.dtype)]


def bundle_arg_count(kernel_dir):
    """Tensor arguments the compiled kernel takes, or None if not readable.

    Counts the ``%arg_N_base_addr`` parameters of ``@sdsc_bundle`` in the
    kernel's ``bundle.mlir``. ``spyre launch`` passes exactly the ``-i`` and
    ``-o`` tensors, so the kernel is launchable from the CLI only if this equals
    their number.
    """
    try:
        text = (Path(kernel_dir) / "bundle.mlir").read_text()
    except OSError:
        return None
    sig = re.search(r"func\.func @sdsc_bundle\(([^)]*)\)", text)
    return len(set(re.findall(r"%arg_(\d+)_base_addr", sig.group(1)))) if sig else None


def conv_fn(name):
    """The op behind example ``name``: conv2d without bias."""
    import torch.nn.functional as F

    ex = EXAMPLES[name]
    return lambda x, weight: F.conv2d(
        x, weight, None, stride=ex.stride, padding=ex.padding
    )


def torch_dtype(name):
    """The torch dtype of example ``name``."""
    import torch

    dtypes = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
    }
    return dtypes[EXAMPLES[name].dtype]


def _active_channels(name):
    """Input channels the ones-input weight sums over (1 for INT8, else all)."""
    return 1 if EXAMPLES[name].dtype == "int8" else C_IN


def make_inputs(name, seed=0):
    """CPU inputs for example ``name``, in its dtype.

    "randn": float dtypes get random normals with a small weight scale so
    outputs stay O(1) and ``atol`` is meaningful; INT8 gets {-1, 0, 1}.
    "ones": all-ones activation; weight is ones on the active channels and
    zero elsewhere.
    """
    import torch

    ex = EXAMPLES[name]
    dtype = torch_dtype(name)
    x_shape, w_shape = ex.inputs
    if ex.init == "ones":
        x = torch.ones(x_shape, dtype=dtype)
        weight = torch.zeros(w_shape, dtype=dtype)
        weight[:, : _active_channels(name)] = 1
        return [x, weight]
    gen = torch.Generator().manual_seed(seed)
    if ex.dtype == "int8":
        x = torch.randint(-1, 2, x_shape, dtype=dtype, generator=gen)
        weight = torch.randint(-1, 2, w_shape, dtype=dtype, generator=gen)
        return [x, weight]
    x = torch.randn(x_shape, generator=gen)
    weight = torch.randn(w_shape, generator=gen) * 0.1
    return [x.to(dtype), weight.to(dtype)]


def op_fn(name):
    """The torch callable the harness compiles for example ``name``."""
    return conv_fn(name)


def expected(name, inputs):
    """CPU result for ``inputs``.

    The ones-input case is built analytically (no conv is run), so it does not
    share a failure mode with the CPU reference. Everything else is a conv
    accumulated in fp32 and cast to the example dtype: CPU fp16/bf16 conv
    accumulates at the input width and is less accurate than the device, and
    CPU has no int8 conv at all.
    """
    import torch

    ex = EXAMPLES[name]
    dtype = torch_dtype(name)
    if ex.init == "ones":
        # In-bounds taps per output position along one axis with padding=1:
        # K - 1 at the first/last index (one tap lands in the zero pad), K
        # elsewhere.
        taps = torch.full((H,), float(K))
        taps[0] = taps[-1] = K - 1
        active = _active_channels(name)
        out = (taps[:, None] * taps[None, :] * active).expand(B, C_OUT, H, W)
        # Anchor the closed form to the three distinct values so a mistake in
        # the construction above cannot make the check vacuous.
        assert out[0, 0, 0, 0] == 4 * active  # corner
        assert out[0, 0, 0, 1] == 6 * active  # edge
        assert out[0, 0, 1, 1] == 9 * active  # interior
        return out.to(dtype)
    return conv_fn(name)(*[t.float() for t in inputs]).to(dtype)


def _compile_one(name, out):
    # Runs in a child process whose TORCHINDUCTOR_CACHE_DIR is already set.
    import torch

    inputs = make_inputs(name)
    fn = torch.compile(conv_fn(name), backend="inductor")
    got = fn(*[t.to("spyre") for t in inputs]).cpu()
    ex = EXAMPLES[name]
    if tuple(got.shape) != ex.output:
        sys.exit(f"{name}: output shape {tuple(got.shape)} != {ex.output}")
    want = expected(name, inputs)
    if not torch.allclose(got.float(), want.float(), atol=ex.atol, rtol=ex.rtol):
        diff = (got.float() - want.float()).abs()
        sys.exit(
            f"{name}: torch.compile result does not match CPU "
            f"(max abs diff {diff.max().item():g}, "
            f"{int((diff > ex.atol).sum())} elements differ)"
        )

    cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    kernels = glob.glob(os.path.join(cache, "inductor-spyre", "*", "spyreCodeDir"))
    if len(kernels) != 1:
        sys.exit(f"{name}: expected 1 kernel, torch.compile produced {len(kernels)}")

    dest = Path(out) / name
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.dirname(kernels[0]), dest)
    print(f"{name}: wrote {dest}")
    if not ex.launchable:
        print(f"{name}: {ex.dtype} is not a spyre launch tensor type; not launchable")
        return
    wanted = len(ex.inputs) + 1  # spyre launch passes the inputs, then one output
    actual = bundle_arg_count(dest)
    if actual is not None and actual != wanted:
        print(
            f"{name}: spyre launch cannot run this kernel: it takes {actual} "
            f"tensors and the CLI passes {wanted}. Run: "
            f"python test_conv_ops.py --launch {name}"
        )
        return
    print(f"{name}: spyre launch {' '.join(cli_args(name))} {dest}")


def sdk_launch(name, dev, path):
    """Launch the compiled kernel through the SDK; return ``(got, runner)``.

    The four tensors are the ones the generated wrapper passes to ``.run``:
    activation, a scalar zero, weight, and the channel-last output (a plain
    ``torch.empty`` would get the default layout, which is not what the kernel
    writes). ``got`` is the NCHW result read back to the host; keep ``runner``
    alive until it has been read.
    """
    import torch
    import spyre_cli
    from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout
    from torch_spyre._inductor.op_spec import spyre_constant_tensor

    ex = EXAMPLES[name]
    dtype = torch_dtype(name)
    device = torch.device("spyre:0")
    x, weight = dev
    zero = spyre_constant_tensor(0, device, dtype)

    _, c_out, h_out, w_out = ex.output
    out = spyre_empty_with_layout(
        (B, h_out, w_out, c_out),
        (h_out * w_out * c_out, w_out * c_out, c_out, 1),
        dtype,
        SpyreTensorLayout(
            device_size=[h_out, w_out, c_out // 64, 1, 64],
            stride_map=[w_out * c_out, c_out, 64, -1, 1],
            device_dtype=DataFormats.SEN169_FP16,
            valid_elements={(0,): h_out, (1,): w_out, (2, 4): c_out, (3,): 1},
        ),
        device=device,
    )
    runner = spyre_cli.launch(x, zero, weight, out, path=path)
    return out.cpu().permute(0, 3, 1, 2), runner  # channel-last -> NCHW


def launch_one(name, kernel_dir):
    """Run the compiled kernel through the SDK and compare with the CPU."""
    import torch

    ex = EXAMPLES[name]
    inputs = make_inputs(name)
    dev = [t.to("spyre") for t in inputs]
    got, runner = sdk_launch(name, dev, kernel_dir)
    del runner  # only after the outputs are read back
    want = expected(name, inputs)
    diff = (got.float() - want.float()).abs()
    if not torch.allclose(got.float(), want.float(), atol=ex.atol, rtol=ex.rtol):
        sys.exit(
            f"{name}: launch result does not match CPU "
            f"(max abs diff {diff.max().item():g}, "
            f"{int((diff > ex.atol).sum())} elements differ)"
        )
    print(f"{name}: launch PASS (matches CPU, max abs diff {diff.max().item():g})")


def generate(names, out=DEFAULT_OUT):
    """Compile ``names`` into ``out``; return the names that failed.

    An example with an ``xfail`` reason is expected to fail: that is reported
    as XFAIL and not returned, and its traceback is hidden (rerun it by name to
    see it). If it compiles after all it is reported as XPASS (and still not
    returned).
    """
    os.makedirs(out, exist_ok=True)
    failed = []
    for name in names:
        xfail = EXAMPLES[name].xfail
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=cache)
            cmd = [sys.executable, __file__, "--one", name, "--out", str(out)]
            stderr = subprocess.DEVNULL if xfail else None
            ok = not subprocess.run(
                cmd, env=env, cwd=HERE, stderr=stderr, check=False
            ).returncode
        if xfail and not ok:
            print(f"{name}: XFAIL (expected) -- {xfail}")
        elif xfail:
            print(f"{name}: XPASS -- expected to fail ({xfail}) but compiled")
        elif not ok:
            failed.append(name)
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Compile WK-MAT-05 conv2d spyre-cli kernels."
    )
    parser.add_argument("names", nargs="*", metavar="name")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument(
        "--launch",
        metavar="name",
        help="run the compiled kernel DIR/<name> through the spyre_cli SDK "
        "and compare with the CPU",
    )
    args = parser.parse_args()

    if args.one:
        _compile_one(args.one, args.out)
        return
    if args.launch:
        if args.launch not in EXAMPLES:
            parser.error(f"unknown example: {args.launch}")
        launch_one(args.launch, Path(args.out) / args.launch)
        return
    unknown = sorted(set(args.names) - set(EXAMPLES))
    if unknown:
        parser.error(f"unknown example(s): {' '.join(unknown)}")
    failed = generate(args.names or sorted(EXAMPLES), args.out)
    if failed:
        sys.exit(f"failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
