#!/usr/bin/env python3
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>
# Target: Python 3.11 (Raspberry Pi OS Bookworm)
"""Compile an ONNX model to native code with Apache TVM (Relax pipeline).

Replaces the project's original Relay-based conversion scripts. Relay was
removed in TVM 0.26, along with `tvm.micro` and `MakeUnpackedAPI`, so that
approach cannot run on any currently installable TVM. This module uses the
Relax pipeline instead.

Three things had to change versus the Relay recipe, all of them load-bearing:

1. The ONNX graph must be simplified first (onnxsim).  A stock Ultralytics
   export carries Shape/Slice/Concat plumbing that leaves symbolic shapes,
   and Relax's LegalizeOps rejects tensors without a static ShapeExpr.
2. `keep_params_in_input=False`.  With True, initialisers become function
   parameters (Vars), and TVM then reads YOLOv8's `Split` operand as dynamic
   and raises "Dynamic Split not yet supported".  With False they stay
   Constants -- which is also what bakes the weights into the artifact.
3. LegalizeOps -> AnnotateTIROpPattern -> FoldConstant -> FuseOps -> FuseTIR
   must be run explicitly; the default pipeline leaves high-level Relax ops
   that the code generator cannot lower.

Usage:
  python models/convert/tvm_compile.py --onnx models/onnx/yolov8n.onnx \\
      --input-name images --input-shape 1,3,640,640 --out build-tvm/yolov8n
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class OnnxSimplifier:
    """Constant-folds an ONNX graph so every tensor has a static shape."""

    def simplify(self, model: object, input_name: str, shape: Tuple[int, ...]) -> object:
        """Return a simplified copy of model, or the original on failure."""
        import onnxsim

        simplified, ok = onnxsim.simplify(
            model, overwrite_input_shapes={input_name: list(shape)}
        )
        if not ok:
            logger.warning("onnxsim reported failure; using the original graph")
            return model
        logger.info("onnxsim: graph simplified")
        return simplified


class RelaxImporter:
    """Imports a static ONNX graph into a Relax IRModule."""

    def build(self, model: object, input_name: str, shape: Tuple[int, ...]) -> object:
        """Return an IRModule with weights baked in as constants."""
        from tvm.relax.frontend.onnx import from_onnx

        # keep_params_in_input=False is required -- see the module docstring.
        return from_onnx(
            model, shape_dict={input_name: shape}, keep_params_in_input=False
        )


class RelaxLowerer:
    """Lowers high-level Relax operators to fused TIR kernels."""

    def lower(self, mod: object) -> object:
        """Return the module after the explicit legalize/fuse pipeline."""
        import tvm
        from tvm import relax

        pipeline = tvm.ir.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        lowered = pipeline(mod)
        logger.info("lowered to %d TIR functions", len(lowered.functions))
        return lowered


class TvmTargetFactory:
    """Maps a friendly target name to a TVM target dict."""

    # The Raspberry Pi 4 is a Cortex-A72.  Naming it explicitly means the
    # generated kernels are scheduled for the real deployment CPU rather than
    # whatever machine happened to run the conversion.
    TARGETS: Dict[str, Dict[str, str]] = {
        "host": {"kind": "llvm"},
        # num-cores drives how aggressively the tuner parallelises; the Pi 4
        # has 4, and claiming the host's count would produce schedules that
        # oversubscribe the target.
        "arm64": {"kind": "llvm", "mtriple": "aarch64-linux-gnu",
                  "mcpu": "cortex-a72", "num-cores": 4},
        "x86_64": {"kind": "llvm", "mtriple": "x86_64-linux-gnu"},
        "c": {"kind": "c"},
    }

    def create(self, name: str) -> object:
        """Return a tvm.target.Target for the given friendly name."""
        import tvm

        if name not in self.TARGETS:
            raise ValueError(f"unknown target '{name}'; pick one of {list(self.TARGETS)}")
        return tvm.target.Target(self.TARGETS[name])


class RpcSettings:
    """Where to reach the RPC tracker that owns the measurement device."""

    def __init__(self, host: str, port: int, key: str) -> None:
        self.host = host
        self.port = port
        self.key = key


class MetaScheduleTuner:
    """Autotunes a lowered module and returns a scheduled executable.

    The default TVM pipeline emits naive loop nests: the lowered TIR contains
    no parallel and no vectorized annotations at all. Tuning searches for
    tiling, vectorisation and threading, which is where essentially all of
    TVM's performance comes from.

    Budget matters. Each fused kernel needs its own search, so a run that
    cannot afford roughly 64 trials per kernel leaves most of them untouched
    and the untuned tail dominates the total.
    """

    TRIALS_PER_TASK: int = 64
    # Each candidate is timed on the device; a Pi is slow, so give a single
    # measurement room to finish before the runner calls it a timeout.
    RPC_TIMEOUT_SEC: float = 120.0
    RPC_SESSION_TIMEOUT_SEC: float = 300.0

    def tune(self, mod: object, target: object, work_dir: str,
             max_trials: int, rpc: Optional["RpcSettings"] = None) -> object:
        """Return a tuned executable for mod.

        With rpc set, candidates are BUILT here but MEASURED on the remote
        device. That matters: the best tiling depends on cache sizes and core
        count, so a schedule tuned against this machine is tuned for the wrong
        computer. Measuring on the Pi is the only way to get schedules that
        are actually good on the Pi.
        """
        import tvm.s_tir.meta_schedule as ms

        runner = "local"
        if rpc is not None:
            logger.info("measuring on %s via tracker %s:%d",
                        rpc.key, rpc.host, rpc.port)
            runner = ms.runner.RPCRunner(
                rpc_config=ms.runner.RPCConfig(
                    tracker_host=rpc.host,
                    tracker_port=rpc.port,
                    tracker_key=rpc.key,
                    session_timeout_sec=self.RPC_SESSION_TIMEOUT_SEC,
                    session_priority=1,
                ),
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=3, repeat=1, min_repeat_ms=100,
                    enable_cpu_cache_flush=True,
                ),
                alloc_repeat=1,
            )

        logger.info("tuning with a budget of %d trials -- this takes a while",
                    max_trials)
        ms.relax_integration.tune_relax(
            mod=mod, target=target, params=None, work_dir=work_dir,
            max_trials_global=max_trials,
            max_trials_per_task=self.TRIALS_PER_TASK,
            runner=runner,
            strategy="evolutionary",
        )
        # Compile through the same explicit apply that compile_with_database
        # uses. tune_relax's companion, compile_relax, silently discards the
        # schedules it was just handed -- so going through it here would throw
        # away every measurement above and produce an artifact that is byte-
        # for-byte an untuned build, with nothing in the log to say so.
        # tune_relax has already persisted the database to work_dir.
        return TvmCompiler().compile_with_database(mod, target, work_dir)


class TvmCompiler:
    """Compiles a lowered Relax module and writes the deployable artifacts."""

    def compile(self, mod: object, target: object) -> object:
        """Return the compiled VMExecutable."""
        import tvm

        return tvm.compile(mod, target=target)

    def compile_with_database(self, mod: object, target: object,
                              tune_dir: str) -> object:
        """Compile mod using schedules from a previous tuning run.

        Applies the database explicitly rather than going through
        meta_schedule.relax_integration.compile_relax, which silently applies
        nothing: it returns a module that still runs the default schedules, so
        a tuned build measures identical to an untuned one and the tuning
        looks like it failed. Applying the transform directly is what actually
        installs the tuned schedules.
        """
        import tvm
        from tvm import relax
        import tvm.s_tir.meta_schedule as ms

        database = ms.database.JSONDatabase(
            path_workload=f"{tune_dir}/database_workload.json",
            path_tuning_record=f"{tune_dir}/database_tuning_record.json",
            allow_missing=False,
        )
        with target, database:
            scheduled = relax.transform.MetaScheduleApplyDatabase()(mod)
        text = str(scheduled)
        logger.info("applied tuned schedules: %d parallel, %d vectorized loops",
                    text.count("T.parallel"), text.count("T.vectorized"))
        return tvm.compile(scheduled, target=target)

    def write_library(self, executable: object, out_base: Path) -> Path:
        """Export a loadable shared library next to out_base."""
        out_base.parent.mkdir(parents=True, exist_ok=True)
        dest = out_base.with_suffix(".so")
        executable.export_library(str(dest))
        logger.info("wrote %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    def write_c_source(self, executable: object, out_base: Path) -> Optional[Path]:
        """Write the generated C kernels, when the target emitted any."""
        for module in executable.mod.imports:
            if module.kind != "c":
                continue
            source = module.inspect_source()
            dest = out_base.with_suffix(".c")
            dest.write_text(source, encoding="utf-8")
            logger.info("wrote %s (%d bytes)", dest, len(source))
            return dest
        return None


class EmbeddedSourceEmitter:
    """Emits the files that link a TVM model into libcctv.

    The result is genuinely self-contained: no .onnx is parsed and no extra
    shared library is loaded, because the fused kernels and the weights are
    both linked into libcctv itself.

      <name>_kernels.o    the fused TIR kernels, as a relocatable object
      <name>_blob.cc      weights + Relax VM bytecode as a byte array

    Both self-register at load time, which is what `system_lib_prefix` on the
    IRModule buys: it marks the module as a system library so its symbols are
    published to TVM's registry instead of being looked up through dlopen.
    Without that attribute the module reports __tvm_is_system_module = False
    and the VM segfaults trying to resolve its kernels.

    The kernels are emitted by the LLVM backend, not the C backend, and that
    is not a detail. TVM's C backend cannot express vector types at all
    ("Cannot convert type float32x20 to C type") and does not realise
    T.parallel either, so a tuned schedule cannot be compiled through it and
    an untuned one runs about 140x slower than tuned LLVM output. All of
    TVM's performance lives in the LLVM backend.
    """

    def emit(self, executable: object, out_dir: Path, name: str) -> List[Path]:
        """Write the embeddable files; return the paths written."""
        out_dir.mkdir(parents=True, exist_ok=True)
        kernels = self._write_kernels(executable, out_dir, name)
        blob = self._write_blob(executable, out_dir, name)
        return [kernels, blob]

    def _write_kernels(self, executable: object, out_dir: Path, name: str) -> Path:
        module = executable.mod.imports[0]
        if module.kind != "llvm":
            raise RuntimeError(
                f"expected an llvm module to embed, got '{module.kind}'. "
                "The C backend cannot compile tuned schedules.")
        dest = out_dir / f"{name}_kernels.o"
        module.write_to_file(str(dest))
        logger.info("wrote %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    def _write_blob(self, executable: object, out_dir: Path, name: str) -> Path:
        from tvm.runtime import _ffi_api

        # system_lib=True makes the blob register itself at static-init time.
        # An empty symbol prefix keeps the C identifiers unadorned; only one
        # TVM model may be embedded per library because of that.
        source = _ffi_api.ModulePackImportsToC(executable.mod, True, "")
        dest = out_dir / f"{name}_blob.cc"
        dest.write_text(source, encoding="utf-8")
        logger.info("wrote %s (%d bytes)", dest, len(source))
        return dest


class NumericValidator:
    """Checks the compiled model against onnxruntime on random input."""

    ATOL: float = 1e-3
    SEED: int = 0

    def validate(self, onnx_path: Path, executable: object, shape: Tuple[int, ...]) -> bool:
        """Return True when TVM and onnxruntime agree within ATOL."""
        import numpy as np

        rng = np.random.default_rng(self.SEED)
        sample = rng.random(shape, dtype=np.float32)
        reference = self._reference(onnx_path, sample)
        actual = self._tvm_output(executable, sample)
        delta = float(np.abs(actual - reference).max())
        logger.info("max abs diff vs onnxruntime: %.3g", delta)
        return delta <= self.ATOL

    def _reference(self, onnx_path: Path, sample: object) -> object:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        return session.run(None, {name: sample})[0]

    def _tvm_output(self, executable: object, sample: object) -> object:
        import tvm
        from tvm import relax

        vm = relax.VirtualMachine(executable, tvm.cpu())
        result = vm["main"](tvm.runtime.tensor(sample))
        return result.numpy() if hasattr(result, "numpy") else result[0].numpy()


class CompilationPipeline:
    """Runs simplify -> import -> lower -> compile -> validate."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

    def run(self) -> None:
        """Execute the full conversion for the configured arguments."""
        import onnx

        shape = tuple(int(v) for v in self._args.input_shape.split(","))
        model = onnx.load(str(self._args.onnx))
        model = OnnxSimplifier().simplify(model, self._args.input_name, shape)
        executable = self._compile(model, shape)
        self._emit(executable)
        self._validate(executable, shape)

    def _compile(self, model: object, shape: Tuple[int, ...]) -> object:
        mod = RelaxImporter().build(model, self._args.input_name, shape)
        mod = RelaxLowerer().lower(mod)
        target = TvmTargetFactory().create(self._args.target)
        logger.info("compiling for target '%s'", self._args.target)
        if self._args.emit_embed:
            # Mark the module a system library so its symbols self-register
            # with TVM's registry at load time. Without this the linked-in
            # object reports __tvm_is_system_module = False and the VM
            # segfaults resolving kernels.
            mod = mod.with_attr("system_lib_prefix", "")
        if self._args.tune:
            rpc = None
            if self._args.rpc_host:
                rpc = RpcSettings(self._args.rpc_host, self._args.rpc_port,
                                  self._args.rpc_key)
            return MetaScheduleTuner().tune(
                mod, target, self._args.tune_dir, self._args.tune_trials, rpc)
        if self._args.tune_db:
            return TvmCompiler().compile_with_database(
                mod, target, self._args.tune_db)
        return TvmCompiler().compile(mod, target)

    def _emit(self, executable: object) -> None:
        out_base = Path(self._args.out)
        if self._args.emit_embed:
            EmbeddedSourceEmitter().emit(
                executable, Path(self._args.emit_embed), out_base.name
            )
            return
        compiler = TvmCompiler()
        if self._args.emit_c:
            compiler.write_c_source(executable, out_base)
        compiler.write_library(executable, out_base)

    def _validate(self, executable: object, shape: Tuple[int, ...]) -> None:
        if self._args.no_validate or self._args.emit_embed or self._args.target != "host":
            logger.info("skipping numeric validation (cross target or disabled)")
            return
        ok = NumericValidator().validate(Path(self._args.onnx), executable, shape)
        if not ok:
            raise RuntimeError("TVM output diverged from onnxruntime beyond tolerance")
        logger.info("numeric validation PASSED")


class CompileCli:
    """Parses arguments for the TVM compilation pipeline."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Return a configured ArgumentParser."""
        p = argparse.ArgumentParser(description="Compile an ONNX model with TVM Relax")
        p.add_argument("--onnx", required=True, help="Path to the .onnx file")
        p.add_argument("--input-name", default="images", help="Graph input tensor name")
        p.add_argument("--input-shape", default="1,3,640,640", help="Comma-separated")
        p.add_argument("--out", required=True, help="Output path without extension")
        p.add_argument(
            "--target",
            default="host",
            choices=list(TvmTargetFactory.TARGETS),
            help="Compilation target (default: host)",
        )
        p.add_argument("--emit-c", action="store_true", help="Also write C kernels")
        p.add_argument(
            "--emit-embed",
            metavar="DIR",
            help=(
                "Emit <name>_kernels.c, <name>_blob.cc and <name>_register.cc "
                "into DIR for linking straight into libcctv (implies --target c)."
            ),
        )
        p.add_argument("--no-validate", action="store_true", help="Skip the numeric check")
        p.add_argument(
            "--tune",
            action="store_true",
            help=(
                "Autotune with MetaSchedule before emitting. Slow (roughly an "
                "hour for YOLOv8n at 320) but it is where TVM's performance "
                "comes from; the default schedules are naive loop nests."
            ),
        )
        p.add_argument("--tune-dir", default="build-tvm/tune",
                       help="Where to keep the tuning database and logs")
        p.add_argument(
            "--tune-db",
            help=(
                "Reuse a tuning database from a previous --tune run instead of "
                "tuning again. Point it at the --tune-dir that produced it."
            ),
        )
        p.add_argument("--tune-trials", type=int, default=6000,
                       help="Total tuning trial budget (default: 6000)")
        p.add_argument(
            "--rpc-host",
            help=(
                "RPC tracker host. Set this to measure candidates on the real "
                "target instead of this machine -- schedules tuned against "
                "the wrong cache size and core count are not optimal."
            ),
        )
        p.add_argument("--rpc-port", type=int, default=9190,
                       help="RPC tracker port (default: 9190)")
        p.add_argument("--rpc-key", default="rpi4",
                       help="Device key registered with the tracker")
        return p

    def execute(self, argv: Optional[Sequence[str]] = None) -> None:
        """Parse argv and run the pipeline."""
        args = self.build_parser().parse_args(argv)
        CompilationPipeline(args).run()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    CompileCli().execute()


if __name__ == "__main__":
    main()
