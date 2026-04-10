import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

METHOD_NAMES = ["pyscenedetect", "hecate", "transnetv2", "gifdroid", "gpt4o"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], log_file: Path, cwd: Path | None = None, env: Dict[str, str] | None = None) -> int:
    """Run a command, streaming stdout/stderr to ``log_file``."""

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


def run_pyscenedetect(video: Path, out_dir: Path, cli: str) -> int:
    cmd = [cli, "-i", str(video), "detect-content", "split-video", "-o", str(out_dir)]
    return run_command(cmd, out_dir / "log.txt")


def run_hecate(video: Path, out_dir: Path, binary: str) -> int:
    cmd = [binary, "-i", str(video), "--print_shot_info", "--print_keyfrm_info"]
    return run_command(cmd, out_dir / "log.txt")


def run_transnetv2(video: Path, out_dir: Path, script: str, python_bin: str) -> int:
    cmd = [python_bin, script, str(video), "--output", str(out_dir)]
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


def _encode_image_b64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_frames(video: Path, frame_dir: Path, target_fps: float, max_frames: int) -> List[Path]:
    """Sample frames uniformly to target_fps, capped by max_frames."""

    ensure_dir(frame_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native_fps / target_fps)), 1)

    frames: List[Path] = []
    idx = 0
    saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame_path = frame_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)
            saved += 1
        idx += 1

    cap.release()
    return frames


def run_gpt4o(video: Path, out_dir: Path, model: str, api_key: str | None, fps: float, max_frames: int) -> int:
    ensure_dir(out_dir)
    log_file = out_dir / "log.txt"
    if not api_key:
        log_file.write_text("OPENAI_API_KEY not provided; skipping GPT-4o.\n", encoding="utf-8")
        return 1

    frames_dir = out_dir / "frames"
    frames = extract_frames(video, frames_dir, fps, max_frames)
    if not frames:
        log_file.write_text("Failed to extract frames from video.\n", encoding="utf-8")
        return 1

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        log_file.write_text(f"Failed to import openai client: {exc}\n", encoding="utf-8")
        return 1

    prompt = (
        "You are a helpful assistant that detects user actions in GUI recordings. "
        "Given a sequence of ordered frames sampled from the video, return the frame numbers (relative to the original video) "
        "where user interactions cause significant GUI changes. Format: a JSON list of boundary frame indices."
    )

    try:
        client = OpenAI(api_key=api_key)
        content = [{"type": "text", "text": prompt}]
        for fp in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_b64(fp)}", "detail": "low"},
                }
            )

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        content_text = completion.choices[0].message.content
    except Exception as inner_exc:  # pragma: no cover
        log_file.write_text(f"GPT-4o request failed: {inner_exc}\n", encoding="utf-8")
        return 1

    (out_dir / "gpt4o_response.txt").write_text(content_text, encoding="utf-8")
    log_file.write_text("Completed GPT-4o request. See gpt4o_response.txt for output.\n", encoding="utf-8")
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to the GUI recording mp4 file to process.",
    )
    parser.add_argument(
        "--output-root",
        default=Path(__file__).resolve().parent / "runs",
        type=Path,
        help="Where to store per-method outputs.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=METHOD_NAMES,
        help="Single baseline to run.",
    )
    parser.add_argument("--pyscenedetect-cli", default="scenedetect", help="PySceneDetect executable name/path.")
    parser.add_argument("--hecate-bin", default="hecate", help="Path to the hecate binary (distribute/bin/hecate).")
    parser.add_argument(
        "--transnetv2-script",
        default="transnetv2.py",
        help="Path to the TransNetV2 inference script (e.g., TransNetV2/transnetv2.py).",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter to use for TransNetV2/GIFdroid.")
    parser.add_argument("--gifdroid-main", default="main.py", help="Path to GIFdroid main.py entrypoint.")
    parser.add_argument("--gifdroid-utg", type=Path, help="Path to GIFdroid UTG JSON file.")
    parser.add_argument("--gifdroid-artifact", type=Path, help="Path to GIFdroid artifact directory (screenshots).")
    parser.add_argument("--gifdroid-out-name", default="gifdroid_out.json", help="Output filename for GIFdroid results (inside method folder).")
    parser.add_argument("--gpt4o-model", default="gpt-4o", help="OpenAI model name for GPT-4o baseline.")
    parser.add_argument(
        "--gpt4o-api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key; defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument("--gpt4o-fps", type=float, default=3.0, help="Frame sampling rate (frames per second) for GPT-4o.")
    parser.add_argument("--gpt4o-max-frames", type=int, default=120, help="Maximum frames to send to GPT-4o.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have a log.txt file.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    method = args.method

    video = args.video
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    results: List[Tuple[str, Path, int]] = []
    case_name = video.parent.name

    out_dir = args.output_root / method / case_name
    log_file = out_dir / "log.txt"
    if args.skip_existing and log_file.exists():
        results.append((method, video, 0))
    else:
        if method == "pyscenedetect":
            rc = run_pyscenedetect(video, out_dir, args.pyscenedetect_cli)
        elif method == "hecate":
            rc = run_hecate(video, out_dir, args.hecate_bin)
        elif method == "transnetv2":
            rc = run_transnetv2(video, out_dir, args.transnetv2_script, args.python_bin)
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
        elif method == "gpt4o":
            rc = run_gpt4o(video, out_dir, args.gpt4o_model, args.gpt4o_api_key, args.gpt4o_fps, args.gpt4o_max_frames)
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
