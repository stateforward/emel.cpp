#!/usr/bin/env python3
"""Live Cactus/libneedle request benchmark for the needle_graph suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import statistics
import subprocess
import sys
import time
import tempfile
from typing import Any, NoReturn

SCHEMA = "needle_request_bench/v1"
MODEL_ID = "route_w4_qat_cact"
MODEL_RELATIVE_PATH = "tests/models/route-w4-qat.cact"
FIXTURE_ID = "tests/fixtures/cact/needle-heldout-prompts.tsv"
WORKLOAD_ID = "needle_heldout_first4_greedy80_eos_v1"
PROMPT_ROWS = 4
MAX_NEW_TOKENS = 80
THREAD_COUNT = 1
THREAD_CONTRACT = "single_thread"
EMEL_THREAD_COUNT = THREAD_COUNT
EMEL_THREAD_CONTRACT = THREAD_CONTRACT
MAX_BENCH_COUNT = 32
DEFAULT_TIMEOUT_SECONDS = 600
MAX_TIMEOUT_SECONDS = 3600
MODEL_SHA256 = "c7f9eb2c3dc5b52292f8903a22580cd60cea79723e8e9fe5ed8e8e4db9f7778d"
FIXTURE_SHA256 = "2b7ce059b63fd029a684861439afb6e7f0a61e4c6790737e4cbd3ef602d65dc8"
NEEDLE_PACKAGE_VERSION = "2.0.8"
NEEDLE_PACKAGE_TREE_SHA256 = "f7710b88d0a59c92f88a1fc2ce7f374633e15351bc731442900f4bf9763a5dd9"
NEEDLE_PYTHON_SHA256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"

NEEDLE_PACKAGE_INIT_SHA256 = "bf72ccda8516da3879a6c68b7df73607dc5386dd59e9ef03ef6141570e877cf0"
NEEDLE_NATIVE_LIBRARY_SHA256 = "0d2e125f36269067407ca4460f2d01b9371887366e5949243de9f03d0d93bc78"
INJECTION_ENVIRONMENT_VARIABLES = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_FALLBACK_FRAMEWORK_PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
)
WORKER_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "XDG_CACHE_HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_MESSAGES",
    "LC_PAPER",
    "LC_NAME",
    "LC_ADDRESS",
    "LC_TELEPHONE",
    "LC_MEASUREMENT",
    "LC_IDENTIFICATION",
    "NEEDLE_THREADS",
)




def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: needle cactus reference: {message}")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash canonical input {path}: {exc}")
    return digest.hexdigest()


def copy_authenticated_file(source: Path, destination: Path, name: str) -> None:
    try:
        if source.is_symlink() or not source.is_file():
            fail(f"{name} is missing, is not a regular file, or is a symlink: {source}")
        with source.open("rb") as source_file, destination.open("xb") as target:
            shutil.copyfileobj(source_file, target, length=1024 * 1024)
        destination.chmod(stat.S_IRUSR)
    except OSError as exc:
        fail(f"cannot stage {name}: {exc}")


def validate_canonical_input(path: Path, expected_sha256: str, name: str) -> None:
    if not path.is_file():
        fail(f"missing canonical {name}: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        fail(
            f"canonical {name} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
def validate_python_interpreter() -> None:
    try:
        executable = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        fail(f"cannot resolve Python interpreter: {exc}")
    actual_sha256 = sha256_file(executable)
    if actual_sha256 != NEEDLE_PYTHON_SHA256:
        fail(
            "Python interpreter SHA-256 mismatch: expected "
            f"{NEEDLE_PYTHON_SHA256}, got {actual_sha256}"
        )


def validate_canonical_path(path: Path, expected: Path, name: str) -> None:
    try:
        if path.resolve(strict=True) != expected.resolve(strict=True):
            fail(f"canonical {name} path substitution is not allowed: {path}")
    except OSError as exc:
        fail(f"cannot resolve canonical {name} path: {exc}")


def exact_int(value: Any, name: str, *, minimum: int = 0,
              maximum: int = MAX_BENCH_COUNT) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        fail(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def positive_finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        fail(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        fail(f"{name} must be finite and positive")
    return parsed


def required_string(record: dict[str, Any], name: str) -> str:
    value = record.get(name)
    if type(value) is not str or not value:
        fail(f"{name} must be a non-empty string")
    return value

def decode_tsv_prompt(line: str) -> str:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 4:
        fail("malformed heldout TSV row")
    try:
        return bytes.fromhex(fields[3]).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        fail(f"invalid heldout prompt hex: {exc}")


def split_rendered_prompt(prompt: str) -> tuple[str | None, list[Any], str]:
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    tools_start = "<tools>"
    tools_end = "</tools>"
    assistant_suffix = f"{im_end}\n{im_start}assistant\n"
    system: str | None = None
    cursor = 0
    system_prefix = f"{im_start}system\n"
    if prompt.startswith(system_prefix):
        end = prompt.find(im_end, len(system_prefix))
        if end < 0 or not prompt.startswith("\n", end + len(im_end)):
            fail("malformed rendered system prompt")
        system = prompt[len(system_prefix):end]
        cursor = end + len(im_end) + 1
    user_prefix = f"{im_start}user\n{tools_start}"
    if not prompt.startswith(user_prefix, cursor) or not prompt.endswith(assistant_suffix):
        fail("rendered prompt does not match the canonical Needle template")
    tools_begin = cursor + len(user_prefix)
    tools_end_pos = prompt.find(tools_end, tools_begin)
    if tools_end_pos < 0 or not prompt.startswith("\n", tools_end_pos + len(tools_end)):
        fail("rendered prompt has no canonical tools block")
    tools_text = prompt[tools_begin:tools_end_pos]
    query_begin = tools_end_pos + len(tools_end) + 1
    query = prompt[query_begin:-len(assistant_suffix)]
    try:
        tools = json.loads(tools_text)
    except json.JSONDecodeError as exc:
        fail(f"rendered tools JSON is invalid: {exc}")
    if not isinstance(tools, list):
        fail("rendered tools JSON must be a list")
    return system, tools, query


def load_requests(path: Path) -> list[tuple[str | None, list[Any], str]]:
    if not path.is_file():
        fail(f"missing workload fixture: {path}")
    requests: list[tuple[str | None, list[Any], str]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                requests.append(split_rendered_prompt(decode_tsv_prompt(line)))
                if len(requests) == PROMPT_ROWS:
                    break
    if len(requests) != PROMPT_ROWS:
        fail(f"workload fixture has fewer than {PROMPT_ROWS} rows")
    system, tools, _ = requests[0]
    for request_system, request_tools, _ in requests[1:]:
        if request_system != system or request_tools != tools:
            fail("canonical first four requests do not share system/tools")
    return requests


def sha256_python_tree(package_root: Path, *, allow_native_library: Path | None = None) -> str:
    digest = hashlib.sha256()
    sources: list[tuple[str, Path]] = []
    try:
        if package_root.is_symlink() or not package_root.is_dir():
            fail(f"Needle package directory is missing or is a symlink: {package_root}")
        for directory, child_directories, files in os.walk(package_root, followlinks=False):
            directory_path = Path(directory)
            for child in child_directories:
                if (directory_path / child).is_symlink():
                    fail(f"Needle package tree contains a symlink: {directory_path / child}")
            for filename in files:
                source = directory_path / filename
                if source.is_symlink():
                    fail(f"Needle package tree contains a symlink: {source}")
                relative = source.relative_to(package_root)
                if source.suffix == ".py":
                    if not source.is_file():
                        fail(f"Needle Python source is not a regular file: {source}")
                    sources.append((relative.as_posix(), source))
                elif any(filename.endswith(suffix)
                         for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                    if allow_native_library is None or source != allow_native_library:
                        fail(f"Needle package tree contains an unauthenticated extension module: {source}")
                elif source.suffix == ".pyc" and "__pycache__" not in relative.parts:
                    fail(f"Needle package tree contains unauthenticated bytecode: {source}")
        for relative, source in sorted(sources):
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        fail(f"cannot authenticate Needle package tree: {exc}")
    return digest.hexdigest()


def resolve_needle_package(needle_root: Path) -> tuple[Path, Path]:
    try:
        if needle_root.is_symlink():
            fail(f"Needle package root must not be a symlink: {needle_root}")
        resolved_root = needle_root.resolve(strict=True)
    except OSError as exc:
        fail(f"cannot resolve Needle package root: {exc}")
    return resolved_root, resolved_root / "needle"


def validate_needle_package(
        needle_root: Path, *, allow_native_library: Path | None = None) -> Path:
    _, package_root = resolve_needle_package(needle_root)
    actual_tree_sha256 = sha256_python_tree(
        package_root, allow_native_library=allow_native_library)
    if actual_tree_sha256 != NEEDLE_PACKAGE_TREE_SHA256:
        fail(
            "Needle package tree SHA-256 mismatch: expected "
            f"{NEEDLE_PACKAGE_TREE_SHA256}, got {actual_tree_sha256}"
        )
    validate_canonical_input(
        package_root / "__init__.py", NEEDLE_PACKAGE_INIT_SHA256,
        "Needle package __init__.py")
    return package_root


def stage_needle_package(needle_root: Path, staged_root: Path) -> Path:
    _, package_root = resolve_needle_package(needle_root)
    staged_package = staged_root / "needle"
    try:
        staged_package.mkdir(mode=stat.S_IRWXU)
        for directory, child_directories, files in os.walk(
                package_root, followlinks=False):
            directory_path = Path(directory)
            relative_directory = directory_path.relative_to(package_root)
            target_directory = staged_package / relative_directory
            for child in child_directories:
                child_path = directory_path / child
                if child_path.is_symlink():
                    fail(f"Needle package tree contains a symlink: {child_path}")
                (target_directory / child).mkdir(mode=stat.S_IRWXU)
            for filename in files:
                source = directory_path / filename
                relative = source.relative_to(package_root)
                if source.is_symlink():
                    fail(f"Needle package tree contains a symlink: {source}")
                if source.suffix == ".py":
                    copy_authenticated_file(
                        source, staged_package / relative,
                        f"Needle Python source {relative.as_posix()}")
                elif any(filename.endswith(suffix)
                         for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                    fail(f"Needle package tree contains an unauthenticated extension module: {source}")
                elif source.suffix == ".pyc" and "__pycache__" not in relative.parts:
                    fail(f"Needle package tree contains unauthenticated bytecode: {source}")
    except OSError as exc:
        fail(f"cannot stage Needle package tree: {exc}")
    validate_needle_package(staged_root)
    return staged_package


def validate_needle_module_identity(needle: Any, package_root: Path) -> None:
    if getattr(needle, "__version__", None) != NEEDLE_PACKAGE_VERSION:
        fail(
            f"Needle package version mismatch: expected {NEEDLE_PACKAGE_VERSION}, "
            f"got {getattr(needle, '__version__', None)!r}"
        )
    module_file = getattr(needle, "__file__", None)
    if type(module_file) is not str:
        fail("imported Needle package has no source file identity")
    try:
        if Path(module_file).resolve(strict=True) != (package_root / "__init__.py").resolve(strict=True):
            fail("imported Needle package does not match the authenticated package root")
    except OSError as exc:
        fail(f"cannot resolve imported Needle package source: {exc}")
    if not hasattr(needle, "Needle"):
        fail("authenticated needle package has no Needle API")
    if not callable(getattr(needle, "_library_path", None)):
        fail("authenticated needle package has no native library selector")


def select_needle_native_library(
        needle: Any, *, allow_override: bool = False) -> Path:
    override = os.environ.get("NEEDLE_LIB_PATH")
    if override and not allow_override:
        fail("NEEDLE_LIB_PATH is unsupported for canonical needle_graph compare")
    try:
        selected = (Path(override) if override is not None
                    else Path(needle._library_path()).expanduser())
    except Exception as exc:
        fail(f"cannot select Needle native library: {exc}")
    if selected.is_symlink():
        fail(f"Needle native library must not be a symlink: {selected}")
    try:
        library = selected.resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        fail(f"cannot resolve Needle native library: {exc}")
    if not library.is_file():
        fail(f"Needle native library is not a regular file: {library}")
    return library


def validate_needle_native_library(
        needle: Any, *, allow_override: bool = False) -> Path:
    library = select_needle_native_library(
        needle, allow_override=allow_override)
    actual_sha256 = sha256_file(library)
    if actual_sha256 != NEEDLE_NATIVE_LIBRARY_SHA256:
        fail(
            "Needle native library SHA-256 mismatch: expected "
            f"{NEEDLE_NATIVE_LIBRARY_SHA256}, got {actual_sha256}"
        )
    return library


def stage_needle_native_library(needle: Any, staged_package: Path) -> Path:
    library = select_needle_native_library(needle)
    staged_library = staged_package / library.name
    copy_authenticated_file(library, staged_library, "Needle native library")
    actual_sha256 = sha256_file(staged_library)
    if actual_sha256 != NEEDLE_NATIVE_LIBRARY_SHA256:
        fail(
            "Needle native library SHA-256 mismatch: expected "
            f"{NEEDLE_NATIVE_LIBRARY_SHA256}, got {actual_sha256}"
        )
    return staged_library



def import_needle(needle_root: Path, package_root: Path):
    if "needle" in sys.modules:
        fail("needle was imported before canonical package authentication")
    init_path = package_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "needle", init_path, submodule_search_locations=[str(package_root)])
    if spec is None or spec.loader is None:
        fail("authenticated Needle package is not importable")
    needle = importlib.util.module_from_spec(spec)
    sys.modules["needle"] = needle
    sys.dont_write_bytecode = True
    bytecode_root = tempfile.TemporaryDirectory(prefix="emel-needle-pycache-")
    sys.pycache_prefix = bytecode_root.name
    try:
        spec.loader.exec_module(needle)
    except Exception as exc:
        sys.modules.pop("needle", None)
        fail(f"cannot import needle from {needle_root}: {exc}")
    validate_needle_module_identity(needle, package_root)
    validate_needle_package(
        needle_root,
        allow_native_library=(Path(os.environ["NEEDLE_LIB_PATH"]).resolve()
                              if "NEEDLE_LIB_PATH" in os.environ else None))
    return needle

def median(values: list[float], name: str) -> float:
    if not values:
        fail(f"live response has no {name} telemetry")
    checked = [positive_finite_number(value, name) for value in values]
    return positive_finite_number(statistics.median(checked), name)


def median_run_means(run_samples: list[list[float]], name: str) -> float:
    if not run_samples or any(not samples for samples in run_samples):
        fail(f"live response has no {name} telemetry")
    run_means = [
        positive_finite_number(statistics.fmean(
            positive_finite_number(value, name) for value in samples), name)
        for samples in run_samples
    ]
    return median(run_means, name)


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).resolve()
    fixture = Path(args.fixture).resolve()
    needle_root = Path(args.needle_root)
    if getattr(args, "staged", False):
        validate_canonical_input(model, MODEL_SHA256, "model")
        validate_canonical_input(fixture, FIXTURE_SHA256, "fixture")
    else:
        repo_root = Path(__file__).resolve().parents[4]
        validate_canonical_path(model, repo_root / MODEL_RELATIVE_PATH, "model")
        validate_canonical_path(fixture, repo_root / FIXTURE_ID, "fixture")
        validate_canonical_input(model, MODEL_SHA256, "model")
        validate_canonical_input(fixture, FIXTURE_SHA256, "fixture")
    warmup_iterations = exact_int(args.warmup_iterations, "warmup_iterations")
    warmup_runs = exact_int(args.warmup_runs, "warmup_runs")
    iterations = exact_int(args.iterations, "iterations", minimum=1)
    runs = exact_int(args.runs, "runs", minimum=1)
    requests = load_requests(fixture)
    if "NEEDLE_LIB_PATH" in os.environ and not getattr(args, "staged", False):
        fail("NEEDLE_LIB_PATH is unsupported for canonical needle_graph compare")
    allowed_library = (Path(os.environ["NEEDLE_LIB_PATH"]).resolve()
                       if getattr(args, "staged", False) else None)
    package_root = validate_needle_package(
        needle_root, allow_native_library=allowed_library)
    authenticated_root = package_root.parent
    needle = import_needle(authenticated_root, package_root)
    native_library = validate_needle_native_library(
        needle, allow_override=getattr(args, "staged", False))
    validate_needle_package(
        authenticated_root, allow_native_library=native_library)
    system, tools, _ = requests[0]
    os.environ["NEEDLE_THREADS"] = str(THREAD_COUNT)
    try:
        engine = needle.Needle(tools=tools, system=system, weights=str(model))
    except Exception as exc:
        fail(f"Needle initialization failed: {exc}")

    def execute() -> tuple[float, float, float]:
        batch_wall_ns = 0
        prefill: list[float] = []
        decode: list[float] = []
        for _, _, query in requests:
            try:
                engine.reset()
            except Exception as exc:
                fail(f"Needle reset failed: {exc}")
            start = time.perf_counter_ns()
            try:
                response = engine.complete(query, max_new_tokens=MAX_NEW_TOKENS)
            except Exception as exc:
                fail(f"Needle complete failed: {exc}")
            batch_wall_ns += time.perf_counter_ns() - start
            if type(response) is not dict or response.get("success") is not True:
                fail("Needle complete returned failure or malformed response")
            prefill.append(positive_finite_number(
                response.get("prefill_tps"), "Needle prefill_tps"))
            decode.append(positive_finite_number(
                response.get("decode_tps"), "Needle decode_tps"))
        return (
            float(batch_wall_ns) / PROMPT_ROWS,
            statistics.fmean(prefill),
            statistics.fmean(decode),
        )

    for _ in range(warmup_runs):
        for _ in range(warmup_iterations):
            execute()

    wall_samples: list[list[float]] = []
    prefill_samples: list[list[float]] = []
    decode_samples: list[list[float]] = []
    for _ in range(runs):
        run_samples = [execute() for _ in range(iterations)]
        wall_samples.append([sample[0] for sample in run_samples])
        prefill_samples.append([sample[1] for sample in run_samples])
        decode_samples.append([sample[2] for sample in run_samples])

    package_version = NEEDLE_PACKAGE_VERSION
    return {
        "schema": SCHEMA,
        "lane": "reference",
        "backend_id": "cactus.libneedle.native",
        "backend_language": "python_ctypes_native",
        "reference_source": "live",
        "model_id": MODEL_ID,
        "model_path": MODEL_RELATIVE_PATH,
        "fixture_id": FIXTURE_ID,
        "workload_id": WORKLOAD_ID,
        "thread_count": THREAD_COUNT,
        "thread_contract": THREAD_CONTRACT,
        "prompt_rows": PROMPT_ROWS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling_id": "cactus_default_unverified",
        "stop_id": "cactus_default_unverified",
        "warmup_iterations": warmup_iterations,
        "warmup_runs": warmup_runs,
        "iterations": iterations,
        "runs": runs,
        "wall_ns_per_request": median_run_means(wall_samples, "wall"),
        "prefill_tokens_per_second": median_run_means(
            prefill_samples, "prefill_tps"),
        "decode_tokens_per_second": median_run_means(
            decode_samples, "decode_tps"),
        "phase_rate_semantics": "cactus_engine_reported_per_request_median_no_token_counts_noncomparable",
        "needle_package_version": package_version,
        "needle_package_tree_sha256": NEEDLE_PACKAGE_TREE_SHA256,
        "needle_native_library_sha256": NEEDLE_NATIVE_LIBRARY_SHA256,
    }


def parse_emel(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing EMEL output: {path}")
    phases: dict[str, dict[str, str]] = {}
    marker: dict[str, str] | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        fail(f"cannot read EMEL output: {exc}")
    for raw in lines:
        if raw.startswith("# needle_graph:") and f"workload_id={WORKLOAD_ID} " in raw:
            marker = {}
            for token in raw.split()[2:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    if key in marker:
                        fail(f"duplicate EMEL marker key: {key}")
                    marker[key] = value
            continue
        if marker is not None and raw and not raw.startswith("#"):
            fields = raw.split()
            metrics: dict[str, str] = {}
            for token in fields[1:]:
                if "=" not in token:
                    fail("malformed EMEL metric token")
                key, value = token.split("=", 1)
                if key in metrics:
                    fail(f"duplicate EMEL metric key: {key}")
                metrics[key] = value
            phase = marker.get("phase", "")
            if phase in phases:
                fail(f"duplicate EMEL request phase: {phase}")
            phases[phase] = {**marker, **metrics}
            marker = None
    if set(phases) != {"wall", "prefill", "decode"}:
        fail("EMEL output does not contain canonical wall/prefill/decode request rows")
    required_marker_keys = {
        "backend_id", "route", "model_id", "fixture_id", "workload_id",
        "thread_count", "thread_contract", "prompt_rows", "max_new_tokens",
        "sampling_id", "stop_id", "warmup_iterations", "warmup_runs",
        "phase_rate_semantics",
    }
    required_metric_keys = {"ns_per_op", "tokens_per_second", "iter", "runs"}
    allowed_keys = required_marker_keys | required_metric_keys | {
        "lane", "case", "phase", "reference", "backend_id",
        "phase_tokens_per_batch",
    }
    for phase, record in phases.items():
        missing = (required_marker_keys | required_metric_keys) - record.keys()
        if missing:
            fail(f"EMEL {phase} row missing keys: {', '.join(sorted(missing))}")
        extra = set(record) - allowed_keys
        if extra:
            fail(f"EMEL {phase} row has unexpected keys: {', '.join(sorted(extra))}")
    wall = phases["wall"]
    expected_emel = {
        "model_id": MODEL_ID, "fixture_id": FIXTURE_ID,
        "workload_id": WORKLOAD_ID,
        "backend_id": "emel_needle_request_serial", "route": "serial",
        "thread_count": str(EMEL_THREAD_COUNT),
        "thread_contract": EMEL_THREAD_CONTRACT, "prompt_rows": str(PROMPT_ROWS),
        "max_new_tokens": str(MAX_NEW_TOKENS),
        "sampling_id": "greedy_argmax_v1", "stop_id": "eos_v1",
        "phase_rate_semantics": "token_weighted_native_graph_noncomparable",
    }
    metadata_mismatches = [key for key, value in expected_emel.items()
                           if wall[key] != value]
    if metadata_mismatches:
        fail("EMEL metadata mismatch: " + ", ".join(metadata_mismatches))
    consistency_keys = required_marker_keys | {"iter", "runs"}
    for phase, record in phases.items():
        mismatches = [key for key in consistency_keys if record[key] != wall[key]]
        if mismatches:
            fail(f"EMEL {phase} row metadata mismatch: " +
                 ", ".join(sorted(mismatches)))
    try:
        return {
            "schema": SCHEMA,
            "lane": "emel",
            "backend_id": required_string(wall, "backend_id"),
            "backend_language": "cpp",
            "reference_source": "live",
            "model_id": required_string(wall, "model_id"),
            "model_path": MODEL_RELATIVE_PATH,
            "fixture_id": required_string(wall, "fixture_id"),
            "workload_id": required_string(wall, "workload_id"),
            "thread_count": exact_int(int(wall["thread_count"]), "EMEL thread_count", minimum=1),
            "thread_contract": required_string(wall, "thread_contract"),
            "prompt_rows": exact_int(int(wall["prompt_rows"]), "EMEL prompt_rows", minimum=1),
            "max_new_tokens": exact_int(int(wall["max_new_tokens"]), "EMEL max_new_tokens", minimum=1,
                                        maximum=MAX_NEW_TOKENS),
            "sampling_id": required_string(wall, "sampling_id"),
            "stop_id": required_string(wall, "stop_id"),
            "warmup_iterations": exact_int(int(wall["warmup_iterations"]), "EMEL warmup_iterations"),
            "warmup_runs": exact_int(int(wall["warmup_runs"]), "EMEL warmup_runs"),
            "iterations": exact_int(int(wall["iter"]), "EMEL iterations", minimum=1),
            "runs": exact_int(int(wall["runs"]), "EMEL runs", minimum=1),
            "wall_ns_per_request": positive_finite_number(float(wall["ns_per_op"]), "EMEL wall_ns_per_request"),
            "prefill_tokens_per_second": positive_finite_number(
                float(phases["prefill"]["tokens_per_second"]), "EMEL prefill_tokens_per_second"),
            "decode_tokens_per_second": positive_finite_number(
                float(phases["decode"]["tokens_per_second"]), "EMEL decode_tokens_per_second"),
            "phase_rate_semantics": required_string(wall, "phase_rate_semantics"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"EMEL output contains invalid typed values: {exc}")


def validate_reference(record: Any) -> dict[str, Any]:
    if type(record) is not dict:
        fail("reference JSON must be an object")
    required_strings = (
        "schema", "lane", "backend_id", "backend_language", "reference_source",
        "model_id", "model_path", "fixture_id", "workload_id", "thread_contract",
        "sampling_id", "stop_id", "phase_rate_semantics", "needle_package_version",
        "needle_package_tree_sha256", "needle_native_library_sha256",
    )
    for key in required_strings:
        required_string(record, key)
    allowed_keys = set(required_strings) | {
        "thread_count", "prompt_rows", "max_new_tokens", "warmup_iterations",
        "warmup_runs", "iterations", "runs", "wall_ns_per_request",
        "prefill_tokens_per_second", "decode_tokens_per_second",
    }
    extra = set(record) - allowed_keys
    if extra:
        fail("reference JSON has unexpected keys: " + ", ".join(sorted(extra)))
    expected = {
        "schema": SCHEMA, "lane": "reference",
        "backend_id": "cactus.libneedle.native",
        "backend_language": "python_ctypes_native", "reference_source": "live",
        "model_id": MODEL_ID, "model_path": MODEL_RELATIVE_PATH,
        "fixture_id": FIXTURE_ID, "workload_id": WORKLOAD_ID,
        "thread_contract": THREAD_CONTRACT,
        "thread_count": THREAD_COUNT,
        "sampling_id": "cactus_default_unverified",
        "stop_id": "cactus_default_unverified",
        "phase_rate_semantics": "cactus_engine_reported_per_request_median_no_token_counts_noncomparable",
        "needle_package_version": NEEDLE_PACKAGE_VERSION,
        "needle_package_tree_sha256": NEEDLE_PACKAGE_TREE_SHA256,
        "needle_native_library_sha256": NEEDLE_NATIVE_LIBRARY_SHA256,
    }
    mismatches = [key for key, value in expected.items() if record[key] != value]
    if mismatches:
        fail("reference metadata mismatch: " + ", ".join(mismatches))
    for key, minimum, maximum in (
        ("thread_count", 1, MAX_BENCH_COUNT), ("prompt_rows", 1, MAX_BENCH_COUNT),
        ("max_new_tokens", 1, MAX_NEW_TOKENS), ("warmup_iterations", 0, MAX_BENCH_COUNT),
        ("warmup_runs", 0, MAX_BENCH_COUNT), ("iterations", 1, MAX_BENCH_COUNT),
        ("runs", 1, MAX_BENCH_COUNT),
    ):
        exact_int(record.get(key), f"reference {key}", minimum=minimum, maximum=maximum)
    for key in ("wall_ns_per_request", "prefill_tokens_per_second",
                "decode_tokens_per_second"):
        positive_finite_number(record.get(key), f"reference {key}")
    return record


def compare(args: argparse.Namespace) -> None:
    emel = parse_emel(Path(args.emel_input))
    try:
        reference = validate_reference(
            json.loads(Path(args.reference_input).read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read reference JSON: {exc}")
    contract_keys = (
        "schema", "model_id", "model_path", "fixture_id", "workload_id",
        "thread_count", "thread_contract", "prompt_rows", "max_new_tokens",
        "warmup_iterations", "warmup_runs", "iterations", "runs",
    )
    mismatches = [key for key in contract_keys if emel.get(key) != reference.get(key)]
    if mismatches:
        fail("lane contract mismatch: " + ", ".join(mismatches))
    print(
        f"needle/request/{WORKLOAD_ID}/emel_wall_diagnostic "
        f"ns_per_request={emel['wall_ns_per_request']:.3f} "
        "timed_scope=pretokenized_native_graph_init_excluded comparable=false"
    )
    print(
        f"needle/request/{WORKLOAD_ID}/cactus_wall_diagnostic "
        f"ns_per_request={reference['wall_ns_per_request']:.3f} "
        "timed_scope=complete_raw_query_public_api comparable=false"
    )
    print(
        f"needle/request/{WORKLOAD_ID}/emel_prefill_diagnostic "
        f"tokens_per_second={emel['prefill_tokens_per_second']:.3f} "
        f"semantics={emel['phase_rate_semantics']} comparable=false"
    )
    print(
        f"needle/request/{WORKLOAD_ID}/cactus_prefill_diagnostic "
        f"tokens_per_second={reference['prefill_tokens_per_second']:.3f} "
        f"semantics={reference['phase_rate_semantics']} comparable=false"
    )
    print(
        f"needle/request/{WORKLOAD_ID}/emel_decode_diagnostic "
        f"tokens_per_second={emel['decode_tokens_per_second']:.3f} "
        f"semantics={emel['phase_rate_semantics']} comparable=false"
    )
    print(
        f"needle/request/{WORKLOAD_ID}/cactus_decode_diagnostic "
        f"tokens_per_second={reference['decode_tokens_per_second']:.3f} "
        f"semantics={reference['phase_rate_semantics']} comparable=false"
    )
    print(
        f"# needle_request_contract: reference=live_cactus_native "
        f"model_id={MODEL_ID} fixture_id={FIXTURE_ID} "
        f"workload_id={WORKLOAD_ID} thread_count={THREAD_COUNT} "
        f"thread_contract={THREAD_CONTRACT} max_new_tokens={MAX_NEW_TOKENS} "
        f"needle_package_version={reference['needle_package_version']} "
        f"needle_package_tree_sha256={reference['needle_package_tree_sha256']} "
        f"needle_native_library_sha256={reference['needle_native_library_sha256']} "
        "wall_comparison=noncomparable_public_api_boundary_mismatch "
        "sampling_stop_output_equivalence=unverified_cactus_public_api "
        f"emel_sampling_id={emel['sampling_id']} emel_stop_id={emel['stop_id']} "
        f"cactus_sampling_id={reference['sampling_id']} cactus_stop_id={reference['stop_id']}"
    )


def bounded_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0 or parsed > MAX_BENCH_COUNT:
        raise argparse.ArgumentTypeError(f"must be in [1, {MAX_BENCH_COUNT}]")
    return parsed


def bounded_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0 or parsed > MAX_BENCH_COUNT:
        raise argparse.ArgumentTypeError(f"must be in [0, {MAX_BENCH_COUNT}]")
    return parsed

def bounded_timeout_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0 or parsed > MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be in [1, {MAX_TIMEOUT_SECONDS}]")
    return parsed

def worker_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in WORKER_ENVIRONMENT_ALLOWLIST
        if name in os.environ
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        environment.update(extra)
    return environment



def run_reference_subprocess(args: argparse.Namespace) -> None:
    if os.name != "posix":
        fail("authenticated worker loader contract is unsupported on this platform")
    timeout_seconds = exact_int(
        args.timeout_seconds, "timeout_seconds", minimum=1,
        maximum=MAX_TIMEOUT_SECONDS)
    output = Path(args.output)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with tempfile.TemporaryDirectory(prefix="emel-needle-auth-") as staging_name:
            staging_root = Path(staging_name)
            staged_model = staging_root / "model.cact"
            repo_root = Path(__file__).resolve().parents[4]
            validate_canonical_path(
                Path(args.model), repo_root / MODEL_RELATIVE_PATH, "model")
            validate_canonical_path(
                Path(args.fixture), repo_root / FIXTURE_ID, "fixture")
            staged_fixture = staging_root / "fixture.tsv"
            staged_needle_root = staging_root / "package"
            staged_needle_root.mkdir(mode=stat.S_IRWXU)
            copy_authenticated_file(Path(args.model), staged_model, "canonical model")
            copy_authenticated_file(Path(args.fixture), staged_fixture, "canonical fixture")
            staged_package = stage_needle_package(
                Path(args.needle_root), staged_needle_root)
            needle = import_needle(staged_needle_root, staged_package)
            staged_library = stage_needle_native_library(needle, staged_package)
            os.environ["NEEDLE_LIB_PATH"] = str(staged_library)
            try:
                validate_canonical_input(staged_model, MODEL_SHA256, "model")
                validate_canonical_input(staged_fixture, FIXTURE_SHA256, "fixture")
                validate_needle_package(
                    staged_needle_root, allow_native_library=staged_library)
                command = [
                    sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                    "run-reference-worker", "--staged",
                    "--model", str(staged_model),
                    "--fixture", str(staged_fixture),
                    "--needle-root", str(staged_needle_root),
                    "--warmup-iterations", str(args.warmup_iterations),
                    "--warmup-runs", str(args.warmup_runs),
                    "--iterations", str(args.iterations), "--runs", str(args.runs),
                    "--output", args.output,
                ]
                completed = subprocess.run(
                    command, check=False, timeout=timeout_seconds,
                    env=worker_environment({"NEEDLE_LIB_PATH": str(staged_library)}))
            finally:
                os.environ.pop("NEEDLE_LIB_PATH", None)
                for name in tuple(sys.modules):
                    if name == "needle" or name.startswith("needle."):
                        sys.modules.pop(name, None)
    except subprocess.TimeoutExpired:
        temporary.unlink(missing_ok=True)
        fail(f"Needle reference process exceeded {timeout_seconds}s timeout")
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise SystemExit(completed.returncode)


def add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--needle-root", default=os.environ.get("EMEL_BENCH_NEEDLE_ROOT", ""))
    parser.add_argument("--warmup-iterations", type=bounded_nonnegative_int, default=1)
    parser.add_argument("--warmup-runs", type=bounded_nonnegative_int, default=1)
    parser.add_argument("--iterations", type=bounded_positive_int, default=1)
    parser.add_argument("--runs", type=bounded_positive_int, default=3)
    parser.add_argument("--output", required=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-reference")
    add_reference_arguments(run)
    run.add_argument("--timeout-seconds", type=bounded_timeout_int,
                     default=DEFAULT_TIMEOUT_SECONDS)
    worker = subparsers.add_parser("run-reference-worker")
    add_reference_arguments(worker)
    worker.add_argument("--staged", action="store_true", help=argparse.SUPPRESS)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--emel-input", required=True)
    compare_parser.add_argument("--reference-input", required=True)
    args = parser.parse_args()
    validate_python_interpreter()
    if args.command in ("run-reference", "run-reference-worker") and not args.needle_root:
        fail("EMEL_BENCH_NEEDLE_ROOT or --needle-root is required")
    if args.command == "run-reference":
        run_reference_subprocess(args)
    elif args.command == "run-reference-worker":
        record = run_reference(args)
        output = Path(args.output)
        temporary = output.with_name(output.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(output)
        except OSError as exc:
            fail(f"cannot write reference JSON: {exc}")
    else:
        compare(args)


if __name__ == "__main__":
    main()
