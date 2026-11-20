# spyre-cli additional tests

Extra workloads for the `spyre_cli` SDK and `spyre launch` CLI. Each workload
compiles an op with `torch.compile` on a Spyre device, checks the result against
a CPU reference, and launches the compiled kernel through `spyre_cli`.

## Layout

```
additional_tests/
├── base/                                   shared harness
│   ├── workloads_common.py                 Example registry, suite discovery, helpers
│   ├── check.py                            compile_one, check_sdk, bench_one
│   └── run_all.py                          runs every example, prints a summary table
└── matrix-tensor-contraction-workloads/    a workload group
    └── test_conv_ops.py                    WK-MAT-05 conv2d examples (suite conv_ops)
```

A **group** is a folder under `additional_tests/`. A **suite** is either a folder
inside a group that contains `<suite>_examples.py`, or a file `test_<suite>.py`
directly in the group folder. `run_all.py` discovers suites both ways. The conv
workload is the second kind: suite `conv_ops`, from `test_conv_ops.py`.

## Requirements

- A Spyre device with the Spyre software stack installed.
- `torch` and `torch_spyre` in the active environment.
- `spyre_cli` importable. It is not installed by `pip install .` here (the
  `pyproject.toml` has no package list and setuptools rejects the flat layout), so
  point `PYTHONPATH` at the package directory:

```bash
export PYTHONPATH=/path/to/torch-spyre/extensions/spyre-cli
```

## Running the conv workload with `run_all.py`

Run from `additional_tests/`:

```bash
cd /home/ravindersinghrawat/fork-torch-spyre/torch-spyre/extensions/spyre-cli/additional_tests
export PYTHONPATH=/home/ravindersinghrawat/fork-torch-spyre/torch-spyre/extensions/spyre-cli

python ./base/run_all.py --group matrix-tensor-contraction-workloads --list
python ./base/run_all.py --group matrix-tensor-contraction-workloads conv_dl16_test
python ./base/run_all.py --group matrix-tensor-contraction-workloads
```

What each command does:

- `--list`: prints every example in the group with its workload, case ID and
  title. Examples marked `[xfail]` are expected not to compile. Nothing runs on
  the device.
- `... conv_dl16_test`: builds and checks one example. Use it first to confirm the
  setup works.
- `...` with no example names: builds and checks every example in the group, then
  prints a summary table and `N/M examples as expected`. The exit status is 0 only
  if every example is as expected.

Both build and check run on the device. The check step launches each kernel
through the `spyre_cli` Python SDK (`spyre_cli.launch`), not the `spyre` command.
The build step prints a `spyre launch ...` line for information only; it does not
run it. `PYTHONPATH` must point at the `spyre-cli` folder so `spyre_cli` can be
imported.

## Running the conv workload directly

`test_conv_ops.py` can also be run on its own, without the harness:

```bash
# Build (compile) every example; kernels go to matrix-tensor-contraction-workloads/build/
python matrix-tensor-contraction-workloads/test_conv_ops.py

# Build selected examples only
python matrix-tensor-contraction-workloads/test_conv_ops.py conv_dl16_test

# Run a compiled kernel through the spyre_cli SDK and compare with the CPU
python matrix-tensor-contraction-workloads/test_conv_ops.py --launch conv_dl16_test
```

Examples and expected results:

| Example | Precision | Check |
|---|---|---|
| `conv_dl16_test` | fp16 (DL16) | random data vs CPU |
| `conv_test` | bf16 | random data vs CPU |
| `conv_stride2_test` | bf16, stride 2 | random data vs CPU |
| `conv_border_dl16` | fp16 | zero border artifacts (analytic output) |
| `conv_border_bf16` | bf16 | zero border artifacts (analytic output) |
| `conv_int8_test` | int8 | `xfail`: does not compile |
| `conv_border_int8` | int8 | `xfail`: does not compile |

An `xfail` example is expected to fail. It prints `XFAIL` and does not fail the
run. If it compiles, it prints `XPASS`; remove its `xfail` marker in that case.

A passing launch prints `launch PASS (matches CPU, max abs diff ...)`. A
`launch result does not match CPU` message is a numerical mismatch.

## Running the harness (`run_all.py`)

```bash
# List examples in a group
python base/run_all.py --group <group> --list

# Build and check every example in a group
python base/run_all.py --group <group>

# Run selected suites or examples
python base/run_all.py --group <group> <suite-or-example> ...

# Run throughput / format metrics a suite defines
python base/run_all.py --group <group> --metrics --iters 20
```

Options:

- `--group G`: the workload group folder (default `precision-scaling-structured-sparsity-workloads`).
- `--out DIR`: kernel output directory (default `<suite>/build`).
- `--list`: print the examples and exit.
- `--metrics`: run `METRICS` instead of the examples.

Each example runs in a fresh process, so a device fault in one example does not
affect the others. The run exits with status 0 only if every selected example is
as expected. If no suites are found in the group, it exits with an error.

> **Note:** `run_all.py` without `--group` uses the default group,
> `precision-scaling-structured-sparsity-workloads`. That folder is not on every
> branch, so it fails with "no suites found" if it is missing. Always pass `--group`.

## Writing a suite for `run_all.py`

A suite module, either `<group>/<suite>/<suite>_examples.py` or
`<group>/test_<suite>.py`, defines:

| Name | Required | Purpose |
|---|---|---|
| `EXAMPLES` | yes | dict of `Example` (fields: `title`, `workload`, `case`, `inputs`, `output`, `dtype`, `out_dtype`, `tol`, `xfail`) |
| `op_fn(name)` | yes | the torch callable that is compiled |
| `make_inputs(name)` | yes | CPU inputs, in kernel argument order |
| `expected(name, inputs)` | yes | CPU reference result |
| `to_device(name, inputs)` | no | device tensors (default: `.to("spyre")`) |
| `check_extra(name, inputs, got)` | no | extra exact check; returns an error string or `None` |
| `sdk_launch(name, dev, path)` | no | for kernels whose tensors or output layout `spyre_cli.launch` cannot take by default; returns `(got, runner)` |
| `METRICS` | no | `{metric name: fn(mod, out, iters)}` |

Without `sdk_launch`, the SDK check launches the kernel with the inputs followed
by one `torch.empty` output of the example's output shape.

## Notes

- `spyre launch` on the command line has no int8 tensor type, so int8 examples
  cannot be launched from the CLI even when they compile.
- The conv kernels take four tensors (activation, a scalar zero, weight, and a
  channel-last output), so `spyre launch` on the command line cannot run them.
  Use `--launch` in `test_conv_ops.py`, which goes through the SDK.
