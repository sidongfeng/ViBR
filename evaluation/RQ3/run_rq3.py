import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


METHODS = ["v2s", "gifdroid", "adbgpt"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], log_file: Path, cwd: Path | None = None, env: Dict[str, str] | None = None) -> int:
    ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    return proc.returncode


def run_v2s(video: Path, out_dir: Path, cli: str, config: Path | None, extra: List[str]) -> int:
    cmd = [cli]
    if config:
        cmd.append(f"--config={config}")
    cmd.extend(extra or [])
    cmd.append(str(video))
    return run_command(cmd, out_dir / "log.txt")


def run_gifdroid(video: Path, out_dir: Path, main_py: Path, python_bin: str, utg: Path, artifact_dir: Path, out_name: str) -> int:
    if not utg.exists():
        raise FileNotFoundError(f"GIFdroid UTG not found: {utg}")
    if not artifact_dir.exists():
        raise FileNotFoundError(f"GIFdroid artifact dir not found: {artifact_dir}")

    out_json = out_dir / out_name
    cmd = [
        python_bin,
        str(main_py),
        f"--video={video}",
        f"--utg={utg}",
        f"--artifact={artifact_dir}",
        f"--out={out_json}",
    ]
    return run_command(cmd, out_dir / "log.txt")


def run_adbgpt(out_dir: Path, python_bin: str) -> int:
    # Assumes `main.py` is on the PYTHONPATH / current working directory and uses its own configs.
    cmd = [python_bin, "main.py"]
    return run_command(cmd, out_dir / "log.txt")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Path to GUI recording video (mp4).")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset of methods to run (v2s, gifdroid, adbgpt). Default: all.",
    )
    parser.add_argument(
        "--output-root",
        default=Path(__file__).resolve().parent / "runs",
        type=Path,
        help="Root folder for outputs.",
    )

    # V2S
    parser.add_argument("--v2s-cli", default="exec_v2s", help="V2S executable/entrypoint.")
    parser.add_argument("--v2s-config", type=Path, help="Path to V2S config JSON (optional if tool has default).")
    parser.add_argument("--v2s-extra", nargs="*", default=[], help="Additional args passed to V2S before the video path.")

    # GIFdroid
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for GIFdroid/AdbGPT.")
    parser.add_argument("--gifdroid-main", default="main.py", help="Path to GIFdroid main.py entrypoint.")
    parser.add_argument("--gifdroid-utg", type=Path, help="Path to GIFdroid UTG JSON file.")
    parser.add_argument("--gifdroid-artifact", type=Path, help="Path to GIFdroid artifact directory (screenshots).")
    parser.add_argument("--gifdroid-out-name", default="gifdroid_out.json", help="Output filename for GIFdroid results.")

    parser.add_argument("--skip-existing", action="store_true", help="Skip a method if its log.txt already exists.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    video = args.video
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    methods = METHODS if args.methods == "all" else [m.strip().lower() for m in args.methods.split(",")]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}")

    case_name = video.parent.name
    results: List[Tuple[str, Path, int]] = []

    for method in methods:
        out_dir = args.output_root / method / case_name
        log_file = out_dir / "log.txt"
        if args.skip_existing and log_file.exists():
            results.append((method, video, 0))
            continue

        if method == "v2s":
            rc = run_v2s(video, out_dir, args.v2s_cli, args.v2s_config, args.v2s_extra)
        elif method == "gifdroid":
            if not args.gifdroid_utg or not args.gifdroid_artifact:
                raise SystemExit("GIFdroid requires --gifdroid-utg and --gifdroid-artifact.")
            rc = run_gifdroid(
                video,
                out_dir,
                Path(args.gifdroid_main),
                args.python_bin,
                args.gifdroid_utg,
                args.gifdroid_artifact,
                args.gifdroid_out_name,
            )
        elif method == "adbgpt":
            rc = run_adbgpt(out_dir, args.python_bin)
        else:
            rc = 1  # unreachable

        results.append((method, video, rc))

    summary_path = args.output_root / "summary.json"
    ensure_dir(summary_path.parent)
    summary_payload = [
        {"method": method, "video": str(video), "status": "ok" if rc == 0 else "failed"}
        for method, video, rc in results
    ]
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    failures = [r for r in results if r[2] != 0]
    if failures:
        print(f"Completed with {len(failures)} failure(s). See {summary_path}.")
        return 1

    print(f"All methods finished successfully. Summary written to {summary_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
