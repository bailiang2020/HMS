#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a sweep over multiple mask counts using a top-k head coordinate file."
    )
    parser.add_argument("--eval_script", type=str, required=True, help="Path to an existing evaluation shell script.")
    parser.add_argument(
        "--mask_source",
        type=str,
        default="coords",
        choices=["coords", "random"],
        help="Whether to mask heads from a coordinate file or by random sampling.",
    )
    parser.add_argument("--coords_file", type=str, default=None, help="Path to the top-k coordinate txt file.")
    parser.add_argument("--mask_counts", type=int, nargs="+", required=True, help="Mask sizes to evaluate.")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used when --mask_source random.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Qwen3_VL_Custom_mask",
        help="Model type to inject into the evaluation script.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label written into the archive file, e.g. ours_img or multi_text.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name used in the default archive filename instead of the eval script name.",
    )
    parser.add_argument(
        "--archive_file",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to results/mask_sweeps/<label>__<script>__<coords>.csv",
    )
    parser.add_argument(
        "--test_data_path_override",
        type=str,
        default=None,
        help="Optional test_data_path override injected into the evaluation script, useful for fast subset evaluation.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Working directory used when running the evaluation script. Defaults to the current directory.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Deprecated. The script now runs sequentially for clearer progress logs.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help="Comma separated CUDA device ids used for parallel runs, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--master_port_base",
        type=int,
        default=29500,
        help="Base torchrun master port. Each parallel job uses an offset from this value.",
    )
    return parser.parse_args()


def build_default_archive_file(args, workdir: Path) -> Path:
    label = args.label or Path(args.eval_script).stem
    script_stem = args.run_name or Path(args.eval_script).stem
    coords_stem = Path(args.coords_file).stem if args.mask_source == "coords" else f"random_seed{args.random_seed}"
    archive_dir = workdir / "results" / "mask_sweeps"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir / f"{label}__{script_stem}__{coords_stem}.csv"


def rewrite_model_type(script_text: str, model_type: str) -> str:
    replaced, count = re.subn(
        r'^MODEL_TYPE="[^"]*"$',
        f'MODEL_TYPE="{model_type}"',
        script_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Failed to rewrite MODEL_TYPE in the evaluation script.")
    return replaced


def rewrite_runtime_launcher(script_text: str, cuda_device: Optional[str], master_port: Optional[int]) -> str:
    rewritten = script_text

    if cuda_device is not None:
        rewritten, count = re.subn(
            r'CUDA_VISIBLE_DEVICES=\S+',
            f'CUDA_VISIBLE_DEVICES={cuda_device}',
            rewritten,
            count=1,
        )
        if count != 1:
            raise ValueError("Failed to rewrite CUDA_VISIBLE_DEVICES in the evaluation script.")

    if master_port is not None:
        rewritten, count = re.subn(
            r'--master_port\s+\d+',
            f'--master_port {master_port}',
            rewritten,
            count=1,
        )
        if count != 1:
            raise ValueError("Failed to rewrite --master_port in the evaluation script.")

    return rewritten


def rewrite_cli_arg_value(script_text: str, arg_name: str, new_value: str) -> str:
    pattern = rf'({re.escape(arg_name)}\s+)(\"[^\"]*\"|\S+)'
    rewritten, count = re.subn(
        pattern,
        rf'\1"{new_value}"',
        script_text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Failed to rewrite {arg_name} in the evaluation script.")
    return rewritten


def extract_final_report(log_text: str) -> str:
    marker = "Final Evaluation Report:"
    idx = log_text.rfind(marker)
    if idx == -1:
        return ""

    lines = log_text[idx + len(marker):].splitlines()
    collected = []
    started = False
    for line in lines:
        if not started and not line.strip():
            continue
        if line.strip():
            started = True
        if not started:
            continue
        collected.append(line.rstrip())
        if "weighted avg" in line:
            break
    return "\n".join(collected).strip()


def parse_report_metrics(report_text: str) -> dict:
    metrics = {}
    if not report_text:
        return metrics

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("precision"):
            continue

        parts = line.split()
        if parts[0] == "accuracy" and len(parts) >= 3:
            metrics["accuracy"] = float(parts[1])
            metrics["accuracy_support"] = int(float(parts[2]))
            continue

        if len(parts) >= 5:
            name = " ".join(parts[:-4])
            precision, recall, f1, support = parts[-4:]
            prefix = name.replace(" ", "_")
            metrics[f"{prefix}_precision"] = float(precision)
            metrics[f"{prefix}_recall"] = float(recall)
            metrics[f"{prefix}_f1"] = float(f1)
            metrics[f"{prefix}_support"] = int(float(support))

    return metrics


def append_rows_to_csv(csv_path: Path, rows: list[dict]):
    fieldnames = [
        "timestamp",
        "label",
        "eval_script",
        "coords_file",
        "mask_source",
        "random_seed",
        "mask_count",
        "return_code",
        "accuracy",
        "accuracy_support",
        "real_0_precision",
        "real_0_recall",
        "real_0_f1",
        "real_0_support",
        "fake_1_precision",
        "fake_1_recall",
        "fake_1_f1",
        "fake_1_support",
        "macro_avg_precision",
        "macro_avg_recall",
        "macro_avg_f1",
        "macro_avg_support",
        "weighted_avg_precision",
        "weighted_avg_recall",
        "weighted_avg_f1",
        "weighted_avg_support",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_cuda_devices(cuda_devices_arg: Optional[str]) -> list[str]:
    if not cuda_devices_arg:
        return []
    return [device.strip() for device in cuda_devices_arg.split(",") if device.strip()]


def build_result_row(
    label: str,
    eval_script: Path,
    coords_file: Optional[Path],
    mask_source: str,
    random_seed: Optional[int],
    mask_count: int,
    return_code: int,
    metrics: dict,
) -> dict:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "eval_script": str(eval_script),
        "coords_file": str(coords_file) if coords_file is not None else "",
        "mask_source": mask_source,
        "random_seed": random_seed if random_seed is not None else "",
        "mask_count": mask_count,
        "return_code": return_code,
        "accuracy": "",
        "accuracy_support": "",
        "real_0_precision": "",
        "real_0_recall": "",
        "real_0_f1": "",
        "real_0_support": "",
        "fake_1_precision": "",
        "fake_1_recall": "",
        "fake_1_f1": "",
        "fake_1_support": "",
        "macro_avg_precision": "",
        "macro_avg_recall": "",
        "macro_avg_f1": "",
        "macro_avg_support": "",
        "weighted_avg_precision": "",
        "weighted_avg_recall": "",
        "weighted_avg_f1": "",
        "weighted_avg_support": "",
    }
    row.update(metrics)
    return row


def run_single_mask_job(
    mask_count: int,
    job_idx: int,
    script_text: str,
    eval_script: Path,
    coords_file: Optional[Path],
    mask_source: str,
    random_seed: int,
    workdir: Path,
    run_logs_dir: Path,
    label: str,
    cuda_devices: list[str],
    master_port_base: int,
) -> dict:
    cuda_device = None
    if cuda_devices:
        cuda_device = cuda_devices[job_idx % len(cuda_devices)]
    master_port = master_port_base + job_idx
    rewritten_script = rewrite_runtime_launcher(script_text, cuda_device=cuda_device, master_port=master_port)

    env = os.environ.copy()
    if mask_source == "coords":
        assert coords_file is not None
        env["MHA_HEAD_MASK_COORDS_FILE"] = str(coords_file)
        env["MHA_HEAD_MASK_TOPK"] = str(mask_count)
        env.pop("MHA_HEAD_MASK_RANDOM_K", None)
        env.pop("MHA_HEAD_MASK_RANDOM_SEED", None)
    else:
        env.pop("MHA_HEAD_MASK_COORDS_FILE", None)
        env.pop("MHA_HEAD_MASK_TOPK", None)
        env["MHA_HEAD_MASK_RANDOM_K"] = str(mask_count)
        env["MHA_HEAD_MASK_RANDOM_SEED"] = str(random_seed)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as tmp_script:
        tmp_script.write(rewritten_script)
        tmp_script_path = Path(tmp_script.name)

    log_path = run_logs_dir / f"mask_{mask_count}.log"
    combined_chunks = []
    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                ["bash", str(tmp_script_path)],
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                combined_chunks.append(line)
                log_f.write(line)
                log_f.flush()
                print(f"[mask={mask_count}] {line}", end="", flush=True)
            proc.wait()
    finally:
        tmp_script_path.unlink(missing_ok=True)

    combined_log = "".join(combined_chunks)

    report_text = extract_final_report(combined_log)
    metrics = parse_report_metrics(report_text)
    row = build_result_row(
        label=label,
        eval_script=eval_script,
        coords_file=coords_file,
        mask_source=mask_source,
        random_seed=random_seed if mask_source == "random" else None,
        mask_count=mask_count,
        return_code=proc.returncode,
        metrics=metrics,
    )

    return {
        "mask_count": mask_count,
        "return_code": proc.returncode,
        "macro_f1": metrics.get("macro_avg_f1", "NA"),
        "report_found": bool(report_text),
        "log_path": str(log_path),
        "row": row,
        "cuda_device": cuda_device,
        "master_port": master_port,
    }


def main():
    args = parse_args()
    eval_script = Path(args.eval_script).resolve()
    if args.mask_source == "coords":
        if not args.coords_file:
            raise ValueError("--coords_file is required when --mask_source coords.")
        coords_file: Optional[Path] = Path(args.coords_file).resolve()
    else:
        coords_file = Path(args.coords_file).resolve() if args.coords_file else None
    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()
    archive_file = Path(args.archive_file).resolve() if args.archive_file else build_default_archive_file(args, workdir)
    label = args.label or eval_script.stem

    script_text = eval_script.read_text(encoding="utf-8")
    rewritten_script = rewrite_model_type(script_text, args.model_type)
    if args.test_data_path_override:
        rewritten_script = rewrite_cli_arg_value(
            rewritten_script,
            "--test_data_path",
            str(Path(args.test_data_path_override)),
        )
    cuda_devices = parse_cuda_devices(args.cuda_devices)

    print(
        f"[MaskSweep] Starting sweep with {len(args.mask_counts)} mask counts. "
        f"eval_script={eval_script} mask_source={args.mask_source} "
        f"{'coords_file=' + str(coords_file) if coords_file is not None else 'random_seed=' + str(args.random_seed)}",
        flush=True,
    )
    if args.test_data_path_override:
        print(f"[MaskSweep] Overriding test_data_path with {args.test_data_path_override}", flush=True)
    if "--test_only" not in rewritten_script:
        print(
            "[MaskSweep] Warning: the provided script does not contain --test_only. "
            "This sweep may be launching training runs instead of evaluation only.",
            flush=True,
        )

    run_logs_dir = archive_file.parent / f"{archive_file.stem}_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MaskSweep] Logs will be written to {run_logs_dir}", flush=True)
    print(f"[MaskSweep] Summary CSV will be written to {archive_file}", flush=True)
    if args.max_workers != 1:
        print(
            f"[MaskSweep] Note: --max_workers={args.max_workers} is ignored. "
            "The script now runs sequentially.",
            flush=True,
        )

    for job_idx, mask_count in enumerate(args.mask_counts):
        cuda_device = cuda_devices[job_idx % len(cuda_devices)] if cuda_devices else None
        master_port = args.master_port_base + job_idx
        print(
            f"[MaskSweep] Launching k={mask_count} cuda={cuda_device} port={master_port}",
            flush=True,
        )
        result = run_single_mask_job(
            mask_count=mask_count,
            job_idx=job_idx,
            script_text=rewritten_script,
            eval_script=eval_script,
            coords_file=coords_file,
            mask_source=args.mask_source,
            random_seed=args.random_seed,
            workdir=workdir,
            run_logs_dir=run_logs_dir,
            label=label,
            cuda_devices=cuda_devices,
            master_port_base=args.master_port_base,
        )
        append_rows_to_csv(archive_file, [result["row"]])
        print(
            f"[MaskSweep] k={mask_count} return_code={result['return_code']} "
            f"macro_f1={result['macro_f1']} cuda={result['cuda_device']} port={result['master_port']}",
            flush=True,
        )
        print(
            f"[MaskSweep] Appended result for k={mask_count} to {archive_file}",
            flush=True,
        )
        if not result["report_found"]:
            print(
                f"[MaskSweep] Warning: failed to extract final report for k={mask_count}. "
                f"See {result['log_path']}",
                flush=True,
            )

    print(f"[MaskSweep] Saved summary to {archive_file}", flush=True)
    print(f"[MaskSweep] Saved per-run logs to {run_logs_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

# python3 scripts/utils/run_mask_sweep.py \
#   --eval_script scripts/sft/DGM4/multi/train.sh \
#   --coords_file results/DGM4/attn_analysis/multi/Qwen3_VL_2B/lora/multi_sft/0-256/plots/ALL/top200_heatmap_norm_share_text_coords.txt \
#   --mask_counts 10 20 30 40 50 75 100 \
#   --label multi_text \
#   --max_workers 2 \
#   --cuda_devices 1,2 \
#   --master_port_base 29500
