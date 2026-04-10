import math
import pathlib
import re
import xml.etree.ElementTree as ET
from functools import cached_property
from typing import List
import cv2
from io import StringIO


"""
Android UI element parser and screenshot annotator.

Parses Android UI XML dumps, filters elements, and labels them on screenshots
using OpenCV for visualization.
"""

# --- Constants ---
BOUNDS_RE = re.compile(r"\[(\d+),(\d+)]\[(\d+),(\d+)]")
MIN_ELEMENT_DIM = 50
RESIZE_FACTOR = 5
CV2_FONT = cv2.FONT_HERSHEY_DUPLEX
CV2_FONT_SCALE = 1.0
CV2_THICKNESS = 2
CV2_ALPHA = 0.8
COLOR_WHITE = (256, 256, 256)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (85, 85, 85)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_PURPLE = (160, 32, 240)

# --- Core Classes ---
class AndroidElement:
    """Represents a UI element on an Android screen with bounds and text."""
    __slots__ = ("path", "bounds", "text", "__dict__")

    def __init__(self, *, path: str, bounds: tuple[int, int, int, int], text: str) -> None:
        self.path = path
        self.bounds = bounds
        self.text = text

    @cached_property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bounds
        return (x1 + x2) // 2, (y1 + y2) // 2

# --- Parsing & Geometry ---
def parse_bounds(s: str) -> tuple[int, int, int, int]:
    """Parse the bounds string from Android XML to a 4-tuple of integers."""
    match = BOUNDS_RE.match(s)
    if not match:
        raise ValueError(f"Invalid bounds: {s}")
    return tuple(map(int, match.groups()))

def build_path(stack: List[ET.Element]) -> str:
    """Build a unique XML path for the UI element using the 'index' attribute."""
    return "/".join(e.attrib["index"] for e in stack if e.tag != "hierarchy")

def is_overlapping(b1, b2, margin, min_dist) -> bool:
    """
    Returns True if two bounding boxes overlap by a given margin and their centers are close.
    Used to filter out duplicate or redundant elements.
    """
    x_overlap = min(b1[2], b2[2]) - max(b1[0], b2[0])
    y_overlap = min(b1[3], b2[3]) - max(b1[1], b2[1])
    if x_overlap <= margin or y_overlap <= margin:
        return False

    # Check if center points are within minimum distance
    c1 = ((b1[0] + b1[2]) // 2, (b1[1] + b1[3]) // 2)
    c2 = ((b2[0] + b2[2]) // 2, (b2[1] + b2[3]) // 2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) <= min_dist

def parse_xml_string(
    xml: str,
    bound_margin: int,
    min_cent_dist: int,
    clickable_only: bool = False
) -> List[AndroidElement]:
    """
    Parse Android UI XML and return a list of AndroidElement objects.

    Skips elements that overlap too much or don't meet text/resource/clickable criteria.
    If clickable_only is True, only clickable elements are included.
    """
    elements, stack = [], []

    for event, elem in ET.iterparse(StringIO(xml), events=["start", "end"]):
        if event == "start":
            stack.append(elem)

            bounds_str = elem.attrib.get("bounds")
            if not bounds_str:
                continue

            try:
                bounds = parse_bounds(bounds_str)
            except Exception:
                continue  # skip invalid bounds

            if bounds == (0, 0, 0, 0):
                continue

            if any(is_overlapping(bounds, e.bounds, bound_margin, min_cent_dist) for e in elements):
                continue

            clickable = elem.attrib.get("clickable") == "true"
            text = elem.attrib.get("text", "")
            resource_id = elem.attrib.get("resource-id", "")

            if clickable_only and not clickable:
                continue

            if not clickable_only and not (text.strip() or resource_id.strip()):
                continue

            path = build_path(stack)
            elements.append(AndroidElement(path=path, bounds=bounds, text=text.strip()))

        elif event == "end":
            elem.clear()
            if stack:
                stack.pop()

    return elements

# --- Visualization ---
def label_screenshot(
    screenshot_path: pathlib.Path,
    screenshot_dir: str,
    name: str,
    elements: List[AndroidElement],
) -> pathlib.Path:
    """
    Draw rectangles and labels around elements on a screenshot.

    Saves the annotated image to screenshot_dir and returns the path.
    """
    image = cv2.imread(str(screenshot_path))
    overlay = image.copy()

    for e in elements:
        cx, cy = e.center
        # Rectangle size is proportional to element bounds, but with a minimum size.
        w = max(MIN_ELEMENT_DIM, (e.bounds[2] - e.bounds[0]) // RESIZE_FACTOR)
        h = max(MIN_ELEMENT_DIM, (e.bounds[3] - e.bounds[1]) // RESIZE_FACTOR)
        top_left = (cx - w // 2, cy - h // 2)
        bottom_right = (cx + w // 2, cy + h // 2)

        # Draw an outlined rectangle for the UI element.
        cv2.rectangle(overlay, top_left, bottom_right, COLOR_PURPLE, thickness=4)

    # Blend the overlay with the original image for highlight effect.
    blended = cv2.addWeighted(overlay, CV2_ALPHA, image, 1 - CV2_ALPHA, 0)

    for idx, e in enumerate(elements):
        cx, cy = e.center
        w = max(MIN_ELEMENT_DIM, (e.bounds[2] - e.bounds[0]) // RESIZE_FACTOR)
        h = max(MIN_ELEMENT_DIM, (e.bounds[3] - e.bounds[1]) // RESIZE_FACTOR)
        top_left = (cx - w // 2, cy - h // 2)

        text = str(idx)
        (tw, th), _ = cv2.getTextSize(text, CV2_FONT, CV2_FONT_SCALE, CV2_THICKNESS)

        # Draw label text above the top-left corner (or below if too close to top).
        text_x = top_left[0]
        text_y = top_left[1] - 8
        if text_y < th:
            text_y = top_left[1] + th + 8

        cv2.putText(
            blended,
            text,
            (text_x, text_y),
            CV2_FONT,
            CV2_FONT_SCALE,
            COLOR_PURPLE,
            CV2_THICKNESS,
        )

    output_path = pathlib.Path(screenshot_dir) / f"{name}.png"
    cv2.imwrite(str(output_path), blended)
    return output_path