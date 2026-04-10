"""
clip_seg.py — CLIP-based video stable-segment detection.

Splits a video into stable segments by computing cosine similarity
between CLIP image embeddings of consecutive frames.

Usage (standalone):
    python clip_seg.py <video_path> [--threshold 0.95] [--interval 3] [--frame-step 1]

As a library:
    from clip_seg import VideoStableSegmentCLIP

    segmenter = VideoStableSegmentCLIP()
    frames    = segmenter.read_frames_from_video("demo.mp4")
    sim_list  = segmenter.calculate_clip_sim_seq(frames)
    segments  = segmenter.detect_keyframes(sim_list)
"""

import argparse
from itertools import groupby

import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class VideoStableSegmentCLIP:
    """
    Video segmenter based on CLIP frame similarity.

    Splits a video into stable segments using CLIP-based cosine
    similarity between consecutive frames.
    """

    def __init__(
        self,
        stable_sim_threshold: float = 0.95,
        stable_interval_threshold: int = 3,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        """
        Args:
            stable_sim_threshold: Similarity threshold; frames below this
                are considered *unstable* (i.e. a transition is happening).
            stable_interval_threshold: How many frames around an unstable
                frame are also marked unstable (smoothing window).
            model_name: HuggingFace model identifier for CLIP.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.sim_threshold = stable_sim_threshold
        self.interval_threshold = stable_interval_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Frame I/O
    # ------------------------------------------------------------------

    @staticmethod
    def read_frames_from_video(video_path: str, frame_step: int = 1) -> list[Image.Image]:
        """
        Read frames from a video file at the given step interval.

        Returns:
            List of PIL Images (RGB).
        """
        print(f"Reading frames from {video_path} (step={frame_step})...")
        frames: list[Image.Image] = []
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_step == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            idx += 1
            print(f"  frame {idx}", end="\r")
        cap.release()
        print(f"\nTotal frames read: {len(frames)}")
        return frames

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _encode_frames(self, frame_list: list[Image.Image]) -> list[torch.Tensor]:
        """Encode a list of PIL images into L2-normalised CLIP embeddings."""
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for i, frame in enumerate(frame_list):
                inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
                feat = self.model.get_image_features(**inputs)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(feat.squeeze(0).cpu())
                print(f"  Encoded {i + 1}/{len(frame_list)}", end="\r")
        print()
        return embeddings

    def calculate_clip_sim_seq(self, frame_list: list[Image.Image]) -> list[float]:
        """
        Compute a sequence of CLIP cosine-similarities between consecutive frames.

        Args:
            frame_list: List of PIL Images (RGB).
        Returns:
            List of float similarity scores (length = len(frame_list) - 1).
        """
        print("Encoding frames with CLIP...")
        embeddings = self._encode_frames(frame_list)

        print("Computing similarity sequence...")
        sim_list: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i], embeddings[i + 1], dim=0
            )
            sim_list.append(sim.item())
        return sim_list

    # ------------------------------------------------------------------
    # Stable-segment detection
    # ------------------------------------------------------------------

    def _stable_flags(self, sim_sequence: list[float]) -> list[bool]:
        """
        Return a boolean mask over *frames* (same length as sim_sequence).
        True  → frame belongs to a stable region.
        False → frame is in or near a transition.
        """
        n = len(sim_sequence)
        flags = [True] * n
        for i, s in enumerate(sim_sequence):
            if s <= self.sim_threshold:
                lo = max(0, i - self.interval_threshold)
                hi = min(n, i + self.interval_threshold + 1)
                for j in range(lo, hi):
                    flags[j] = False
        return flags

    def detect_keyframes(self, sim_sequence: list[float]) -> list[tuple[int, int]]:
        """
        Detect stable segments from a similarity sequence.

        Args:
            sim_sequence: List of per-frame similarity scores.
        Returns:
            List of (start_frame, end_frame) tuples for each stable segment.
            Within each segment, frames[start] … frames[end] are stable.
        """
        flags = self._stable_flags(sim_sequence)

        segments: list[tuple[int, int]] = []
        idx = 0
        for is_stable, group in groupby(flags):
            length = sum(1 for _ in group)
            if is_stable:
                segments.append((idx, idx + length - 1))
            idx += length

        return segments


# ------------------------------------------------------------------
# CLI entry-point (for standalone testing)
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based video stable-segment detection.",
    )
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Similarity threshold (default: 0.95)")
    parser.add_argument("--interval", type=int, default=3,
                        help="Stable interval threshold (default: 3)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Read every N-th frame (default: 1)")
    args = parser.parse_args()

    segmenter = VideoStableSegmentCLIP(
        stable_sim_threshold=args.threshold,
        stable_interval_threshold=args.interval,
    )

    frames = segmenter.read_frames_from_video(args.video_path, frame_step=args.frame_step)
    sim_list = segmenter.calculate_clip_sim_seq(frames)
    segments = segmenter.detect_keyframes(sim_list)

    print(f"\nFound {len(segments)} stable segments:")
    for i, (s, e) in enumerate(segments):
        print(f"  Segment {i}: frames {s} – {e}")


if __name__ == "__main__":
    main()