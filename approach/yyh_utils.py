import cv2
import numpy as np
from itertools import groupby
from skimage.metrics import structural_similarity as ssim


def extract_Y(img):
    """
    Extracts the Y (luminance) channel from a BGR image.
    Args:
        img (np.ndarray): OpenCV BGR image.
    Returns:
        np.ndarray: Y channel (grayscale).
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, _, _ = cv2.split(img_yuv)
    return y


def read_frames_from_video(video, header_pixel_size):
    """
    Reads all frames from a video file.

    Args:
        video (str): Path to video file.
        header_pixel_size (int): Number of pixels to crop from the top of Y channel for 'y_frames'.

    Returns:
        frames (list): List of original frames (BGR, OpenCV format).
        y_frames (list): List of Y channel frames (cropped at top).
    """
    print("Reading frames from video...")
    frames = []
    y_frames = []
    vidcap = cv2.VideoCapture(video)
    success, frame = vidcap.read()
    frames.append(frame)
    y_frame = extract_Y(frame)
    # cut header (remove top 'header_pixel_size' pixels)
    y_frames.append(y_frame[header_pixel_size:])
    while success:
        success, frame = vidcap.read()
        if not success:
            break
        frames.append(frame)
        y_frame = extract_Y(frame)
        y_frames.append(y_frame[header_pixel_size:])
        print("Reading frame: ", len(frames), end="\r")
    vidcap.release()
    return frames, y_frames


class VideoStableSegment:
    """
    Video segmenter based on frame similarity.

    Splits a video into stable segments using SSIM-based similarity.
    """
    def __init__(self, stable_sim_threshold=0.95, stable_interval_threshold=3):
        """
        Args:
            stable_sim_threshold (float): SSIM threshold below which frames are considered unstable.
            stable_interval_threshold (int): Number of consecutive unstable frames before ending a stable segment.
        """
        self.sim_threshold = stable_sim_threshold
        self.interval_threshold = stable_interval_threshold

    def return_stable_flags(self, list_):
        """
        Returns a boolean list: True where the region is stable, False where not (with interval adjustment).
        """
        result_list = [True for _ in list_]

        for index, item in enumerate(list_):
            if item <= self.sim_threshold:
                start = max(0, index - self.interval_threshold)
                end = min(len(list_), index + self.interval_threshold + 1)

                for inner_index in range(start, end):
                    result_list[inner_index] = False

        return result_list

    def detect_keyframes(self, sim_sequence):
        """
        Detects stable segment keyframes based on similarity sequence.

        Args:
            sim_sequence (list): List of SSIM similarity scores between frames.

        Returns:
            List of (start, end) frame indices for each stable segment.
        """
        stable_flag_list = self.return_stable_flags(sim_sequence)
        stable_flag_list.reverse()

        keyframe_list = []
        keyframe_start = []

        idx = 0
        for k, g in groupby(stable_flag_list):
            if k:
                keyframe_list.append(idx)
            idx += sum(1 for i in g)
            if k:
                keyframe_start.append(idx)  # not that we do not -1 here!

        keyframes_index = [len(stable_flag_list) - x for x in keyframe_list]
        keyframes_index.reverse()

        keyframes_start_index = [len(stable_flag_list) - x for x in keyframe_start]
        keyframes_start_index.reverse()

        return [(a, b) for a, b in zip(keyframes_start_index, keyframes_index)]


def calculate_sim_seq(frame_list):
    """
    Calculate a sequence of SSIM-based similarities between consecutive frames.

    Args:
        frame_list (list): List of frames (Y channel, grayscale numpy arrays).

    Returns:
        sim_list (list): List of SSIM similarity scores (float), length = len(frame_list) - 1.
    """
    print("Computing SSIM similarity sequence...")
    sim_list = []
    for i in range(len(frame_list) - 1):
        score = ssim(frame_list[i], frame_list[i + 1])
        sim_list.append(score)
        print(f"  SSIM {i + 1}/{len(frame_list) - 1}", end="\r")
    print()
    return sim_list