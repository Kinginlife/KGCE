import base64
import hashlib
import io
import os
import re
import subprocess
import time
from collections import Counter
from functools import cache
from typing import Callable, List, Optional, Tuple

import cv2
import easyocr
import imageio as imio
import networkx as nx
import numpy as np
import psutil
import pyperclip
import requests
import torch
from networkx import DiGraph, path_graph
from numpy.linalg import norm
from PIL import Image
import winreg
from kgce import SubTask, TaskGenerator, action, evaluator
from kgce.actions.kgce_actions import check_submit, submit
import tempfile

class ImageMatcher:
    def __init__(self, top_k: int = 4096):
        self.xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k
        )
        self.top_k = top_k

    def warp_corners_and_draw_matches(
        self,
        ref_points: np.ndarray,
        dst_points: np.ndarray,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, mask = cv2.findHomography(
            ref_points,
            dst_points,
            cv2.USAC_MAGSAC,
            3.5,
            maxIters=1000,
            confidence=0.999,
        )
        mask = mask.flatten()

        h, w = img1.shape[:2]
        corners_img1 = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_img1, H)

        img2_with_corners = img2.copy()
        for i in range(len(warped_corners)):
            start_point = tuple(warped_corners[i - 1].astype(int))
            end_point = tuple(warped_corners[i].astype(int))
            cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)

        keypoints1 = [cv2.KeyPoint(p, p, 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p, p, 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

        img_matches = cv2.drawMatches(
            img1,
            keypoints1,
            img2_with_corners,
            keypoints2,
            matches,
            None,
            matchColor=(0, 255, 0),
            flags=2,
        )

        return img_matches, warped_corners

    def _get_bounding_box(self, warped_corners: np.ndarray, img_shape: Tuple[int, int]) -> List[int]:
        h, w = img_shape
        x_min = np.min(warped_corners[:, 0, 0])
        x_max = np.max(warped_corners[:, 0, 0])
        y_min = np.min(warped_corners[:, 0, 1])
        y_max = np.max(warped_corners[:, 0, 1])

        x_min = max(0, x_min)
        x_max = min(w - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(h - 1, y_max)

        return [int(x_min), int(x_max), int(y_min), int(y_max)]

    def _resize_image(self, img1: np.ndarray, img2: np.ndarray, scale: float, match_dimension: str) -> Tuple[np.ndarray, np.ndarray]:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if match_dimension == "height":
            new_height = int(h2 * scale)
            new_width = int(w1 * (new_height / h1))
        elif match_dimension == "width":
            new_width = int(w2 * scale)
            new_height = int(h1 * (new_width / w1))
        else:
            raise ValueError("match_dimension must be either 'height' or 'width'.")

        resized_img1 = cv2.resize(img1, (new_width, new_height))
        return resized_img1, img2

    def get_resizing_functions(self) -> List[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        return [
            lambda x, y: (x, y),
            lambda x, y: self._resize_image(x, y, 1.0, "height"),
            lambda x, y: self._resize_image(x, y, 1.0, "width"),
            lambda x, y: self._resize_image(x, y, 0.5, "height"),
            lambda x, y: self._resize_image(x, y, 0.5, "width"),
        ]

    def match_images(
        self,
        im1_path: str,
        im2_path: str,
        top_k: int = 4096,
        match_num_threshold: int = 80,
    ) -> Tuple[Optional[List[int]], Optional[np.ndarray], int]:
        im1 = self.load_and_convert_image(im1_path)
        im2 = self.load_and_convert_image(im2_path)

        best_matches = {
            "count": 0,
            "im1_resized": None,
            "im2_resized": None,
            "mkpts_0": None,
            "mkpts_1": None,
        }

        for resize_func in self.get_resizing_functions():
            try:
                im1_resized, im2_resized = resize_func(im1, im2)
                mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(im1_resized, im2_resized, top_k=top_k)

                if len(mkpts_0) > best_matches["count"]:
                    best_matches.update(
                        {
                            "count": len(mkpts_0),
                            "im1_resized": im1_resized,
                            "im2_resized": im2_resized,
                            "mkpts_0": mkpts_0,
                            "mkpts_1": mkpts_1,
                        }
                    )
            except Exception:
                continue

        if best_matches["count"] >= match_num_threshold:
            canvas, warped_corners = self.warp_corners_and_draw_matches(
                best_matches["mkpts_0"],
                best_matches["mkpts_1"],
                best_matches["im1_resized"],
                best_matches["im2_resized"],
            )
            bbox = self._get_bounding_box(warped_corners, im2_resized.shape[:2])
        else:
            bbox, canvas = None, None

        return bbox, canvas, best_matches["count"]

    def load_and_convert_image(self, filepath: str) -> np.ndarray:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Unable to load image from {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return image


image_matcher = ImageMatcher()


def from_env_load_and_save_file(env, file_path: str, output_dir: Optional[str] = None) -> str:
    """
    Load a file, convert it to bytes, and save it to a local directory with the same basename.

    Args:
        env: The environment object with the _action_endpoint method.
        file_path (str): The path to the file to be loaded.
        output_dir (Optional[str]): The directory where the file should be saved.
            If None, uses the system's temporary directory.

    Returns:
        str: The path to the saved file.
    """

    @action(env_name="windows")
    def get_encoded_file(file_path: str) -> Optional[bytes]:
        try:
            with open(file_path, "rb") as file:
                file_bytes = file.read()
                encoded_string = base64.b64encode(file_bytes).decode("utf-8")
        except Exception:
            return None

        return encoded_string

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(tempfile.gettempdir(), "local_save")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the file and convert to bytes
    encoded_string = env._action_endpoint(get_encoded_file, {"file_path": file_path})

    if encoded_string is None:
        raise ValueError(f"Failed to load or encode the file: {file_path}")

    # Decode the Base64 string back to bytes
    decoded_bytes = base64.b64decode(encoded_string.encode("utf-8"))

    # Create the output file path
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name)

    # Save the decoded bytes to the output path
    with open(output_file_path, "wb") as file:
        file.write(decoded_bytes)

    return output_file_path


def crop_image(img: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crops the image based on the bounding box coordinates.

    Parameters:
    img (np.ndarray): The input image.
    bbox (List[int]): Bounding box coordinates [x_min, x_max, y_min, y_max].

    Returns:
    np.ndarray: The cropped image.
    """
    x_min, x_max, y_min, y_max = bbox
    return img[y_min:y_max, x_min:x_max]


def calculate_bbox_center(bbox: List[int]) -> Tuple[int, int]:
    """
    Calculates the center of a bounding box.

    Parameters:
    bbox (List[int]): The bounding box coordinates [x_min, x_max, y_min, y_max].

    Returns:
    Tuple[int, int]: The center coordinates (x, y).
    """
    x_min, x_max, y_min, y_max = bbox
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    return x_center, y_center


def is_bbox_in_direction(bbox_1: List[int], bbox_2: List[int], direction: str) -> bool:
    """
    Check if the center of bbox_1 is in the specified direction relative to the center of bbox_2.

    Args:
        bbox_1 (List[int]): The bounding box coordinates [x_min, x_max, y_min, y_max] of the first bounding box.
        bbox_2 (List[int]): The bounding box coordinates [x_min, x_max, y_min, y_max] of the second bounding box.
        direction (str): The direction to check ("left", "right", "above", "below").

    Returns:
        bool: True if the center of bbox_1 is in the specified direction relative to bbox_2, False otherwise.
    """

    center_1 = calculate_bbox_center(bbox_1)
    center_2 = calculate_bbox_center(bbox_2)

    if direction == "left":
        return center_1[0] < center_2[0]
    elif direction == "right":
        return center_1[0] > center_2[0]
    elif direction == "above":
        return center_1[1] < center_2[1]
    elif direction == "below":
        return center_1[1] > center_2[1]
    else:
        raise ValueError("Invalid direction. Use 'left', 'right', 'above', or 'below'.")


def ocr_text_matching(
    image_path: str, text: str
) -> Optional[Tuple[List[int], str, float]]:
    """
    Performs OCR on an image to find a specific text string and returns the bounding box, matched text, and confidence level.

    Parameters:
    image_path (str): The path to the image file.
    text (str): The text string to search for in the image.

    Returns:
    Optional[Tuple[List[int], str, float]]: The bounding box coordinates [x_min, y_min, x_max, y_max], the matched text, and the confidence level if found, otherwise None.
    """
    reader = easyocr.Reader(["en"])
    result = reader.readtext(image_path)

    for entry in result:
        bbox, detected_text, confidence = entry
        if text in detected_text:
            # Extract the bounding box coordinates
            x_min = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
            x_max = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
            y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            return (
                [int(x_min), int(x_max), int(y_min), int(y_max)],
                detected_text,
                confidence,
            )

    return None


def convert_file_to_images(file_path: str) -> List[str]:
    """
    Convert a file to JPG images using LibreOffice and return the list of image file paths.

    Args:
        file_path (str): The path to the file.

    Returns:
        List[str]: List of paths to the generated image files.
    """
    # Set LibreOffice path for Windows
    libreoffice_path = "C:\\Program Files\\LibreOffice\\program\\soffice.exe"

    # Set output format and directory
    output_format = "jpg"
    output_dir = os.path.join(tempfile.gettempdir(), "converted_images")
    os.makedirs(output_dir, exist_ok=True)

    # Run LibreOffice conversion command
    result = subprocess.run(
        [
            libreoffice_path,
            "--headless",
            "--convert-to",
            output_format,
            "--outdir",
            output_dir,
            file_path,
        ],
        capture_output=True,
        text=True,
    )

    # Check if the conversion was successful
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    # Collect the generated image file paths
    image_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(f".{output_format}")
    ]

    # Verify if the files were successfully saved
    if not image_files:
        raise FileNotFoundError(
            f"No {output_format} files found in the output directory"
        )

    # Get the basename of the original file (without extension)
    file_basename = os.path.splitext(os.path.basename(file_path))

    # Check if any of the images match the basename of the original file
    matching_images = [f for f in image_files if file_basename in os.path.basename(f)]
    if not matching_images:
        raise FileNotFoundError(
            f"No images found with basename matching the original file: {file_basename}"
        )

    return matching_images


def cleanup_files(files: List[str]):
    """
    Delete the list of files.

    Args:
        files (List[str]): List of paths to the files to be deleted.
    """
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except PermissionError as e:
            print(f"Failed to delete {file}: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {file}")
        except Exception as e:
            print(f"An error occurred while deleting {file}: {e}")


def is_valid_url(url: str) -> bool:
    """
    Check if the string is a valid HTTP/HTTPS URL.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    url_pattern = re.compile(
        r"^(https?://)"  # http:// or https://
        r"(?:"  # Domain or IP
        r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?"  # Domain
        r"|localhost"  # localhost
        r"|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"  # IPv4
        r"|$$[A-F0-9:.]+$$"  # IPv6
        r")"
        r"(?::\d+)?"  # Optional port
        r"(?:/?|[/?]\S+)$",  # Path and query
        re.IGNORECASE,
    )
    return bool(re.match(url_pattern, url))


def is_valid_image_data_uri(uri):
    # Regular expression to check if the string is a valid Data URI for image formats
    data_uri_pattern = re.compile(
        r"^data:image/(png|jpeg|gif|svg\+xml|bmp|webp);base64,[A-Za-z0-9+/]+={0,2}$",
        re.IGNORECASE,
    )
    return bool(re.match(data_uri_pattern, uri))


def is_github_repo_url(url):
    # Regular expression to check if the URL is a GitHub repository URL
    github_repo_pattern = re.compile(
        r"^https?://"  # Protocol
        r"github\.com/"  # Domain
        r"[^/]+/"  # Username
        r"[^/]+/?$",  # Repository name, optional trailing slash
        re.IGNORECASE,
    )
    return bool(re.match(github_repo_pattern, url))


def get_rgb_values_outside_bbox(
    img: np.ndarray, bbox: List[int], margin: int = 10
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Reads the pixel color RGB values outside of the bounding box with an additional margin and finds the most frequent RGB value.

    Parameters:
    img (np.ndarray): The input image.
    bbox (List[int]): Bounding box coordinates [x_min, x_max, y_min, y_max].
    margin (int): The margin to add outside the bounding box. Default is 10.

    Returns:
    Tuple[np.ndarray, Tuple[int, int, int]]: The RGB values outside the bounding box with the margin and the most frequent RGB value.
    """
    x_min, x_max, y_min, y_max = bbox

    # Ensure the coordinates with margin are within image dimensions
    x_min_with_margin = max(0, x_min - margin)
    x_max_with_margin = min(img.shape[1], x_max + margin)
    y_min_with_margin = max(0, y_min - margin)
    y_max_with_margin = min(img.shape[0], y_max + margin)

    # Create a mask for the bounding box area with margin
    mask = np.ones(img.shape[:2], dtype=bool)
    mask[y_min_with_margin:y_max_with_margin, x_min_with_margin:x_max_with_margin] = (
        False
    )

    # Extract the RGB values outside the bounding box with margin
    rgb_values = img[mask]

    # Find the most frequent RGB value
    rgb_values_tuple = [tuple(rgb) for rgb in rgb_values]
    most_common_rgb = Counter(rgb_values_tuple).most_common(1)[0][0]

    return list(most_common_rgb)[::-1]


def contains_required_strings(clipboard_content: str, required_strings: list) -> bool:
    """
    Check if all required strings are present in the clipboard content.

    Args:
        clipboard_content (str): The content from the clipboard.
        required_strings (list): A list of required strings to check.

    Returns:
        bool: True if all required strings are found in the clipboard content, False otherwise.
    """
    for string in required_strings:
        if string not in clipboard_content:
            return False
    return True


@evaluator(env_name="windows")
def verify_file_content_with_clipboard(file_path: str) -> bool:
    """
    Verify that the content of the file matches the clipboard content line by line.

    Args:
        file_path (str): The path to the file to verify.

    Returns:
        bool: True if the file content matches the clipboard content, False otherwise.
    """

    def verify_content_with_clipboard(file_content: str) -> bool:
        """
        Verify that the provided file content matches the clipboard content line by line.

        Args:
            file_content (str): The content of the file to verify.

        Returns:
            bool: True if the file content matches the clipboard content, False otherwise.
        """
        clipboard_content = pyperclip.paste()
        clipboard_lines = clipboard_content.split("\n")
        file_lines = file_content.split("\n")

        # Check if each line from the clipboard content is in the corresponding line in the file content
        for clipboard_line, file_line in zip(clipboard_lines, file_lines):
            if clipboard_line not in file_line:
                return False

        return True

    with open(file_path, "r") as file:
        file_content = file.read()

    return verify_content_with_clipboard(file_content)


@evaluator(env_name="windows")
def verify_odt_file_content_with_clipboard(file_path: str) -> bool:
    """
    Verify that the content of the ODT file matches the clipboard content.

    Args:
        file_path (str): The path to the ODT file to verify.

    Returns:
        bool: True if the ODT file content matches the clipboard content, False otherwise.
    """
    from odf import teletype, text
    from odf.opendocument import load

    @evaluator(env_name="windows")
    def verify_odt_file_content_with_clipboard(file_path: str) -> bool:
        """
        Verify that the content of the ODT file matches the clipboard content.

        Args:
            file_path (str): The path to the ODT file to verify.

        Returns:
            bool: True if the ODT file content matches the clipboard content, False otherwise.
        """

        def verify_content_with_clipboard(file_content: str) -> bool:
            """
            Verify that the provided file content matches the clipboard content.

            Args:
                file_content (str): The content of the file to verify.

            Returns:
                bool: True if the file content matches the clipboard content, False otherwise.
            """
            clipboard_content = pyperclip.paste()

            # Normalize both contents by stripping whitespace and splitting into lines
            clipboard_lines = [line.strip() for line in clipboard_content.split("\n") if line.strip()]
            file_lines = [line.strip() for line in file_content.split("\n") if line.strip()]

            # Check if all clipboard lines are present in the file content
            return all(clipboard_line in file_content for clipboard_line in clipboard_lines)

        # Load the ODT file and extract text content
        textdoc = load(file_path)
        allparas = textdoc.getElementsByType(text.P)
        odt_content = "\n".join([teletype.extractText(p) for p in allparas])

        # Verify the content with clipboard
        return verify_content_with_clipboard(odt_content)


@evaluator(env_name="windows", local=True)
def verify_combined_image(
    image_path_1: str, image_path_2: str, file_path: str, direction: str, env
) -> bool:
    """
    Check if the combined file contains both input images without overlay and in the specified direction.

    Args:
        image_path_1 (str): Path to the first image.
        image_path_2 (str): Path to the second image.
        file_path (str): Path to the combined file.
        direction (str): The direction to check ("left", "right", "above", "below").

    Returns:
        bool: True if the combined file contains both input images in the specified direction without overlay, False otherwise.
    """

    saved_image_path_1 = from_env_load_and_save_file(env, image_path_1)
    saved_image_path_2 = from_env_load_and_save_file(env, image_path_2)
    saved_file_path = from_env_load_and_save_file(env, file_path)

    # Determine if file_path is already an image

    if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        combined_image_path = saved_file_path
    else:
        # Convert the file to images
        combined_image_path = convert_file_to_images(saved_file_path)[0]

    try:
        # Match the first image within the combined image
        bbox_1, _, _ = image_matcher.match_images(
            saved_image_path_1, combined_image_path
        )

        # Match the second image within the combined image
        bbox_2, _, _ = image_matcher.match_images(
            saved_image_path_2, combined_image_path
        )

        # Check if both bounding boxes are found
        if bbox_1 is None or bbox_2 is None:
            return False

        # Check if bbox_1 is in the specified direction relative to bbox_2
        correct_direction = is_bbox_in_direction(bbox_1, bbox_2, direction)

        return correct_direction
    finally:
        # Cleanup intermediate image files if they were created
        cleanup_files(
            [
                combined_image_path,
                saved_image_path_1,
                saved_image_path_2,
                saved_file_path,
            ]
        )


@evaluator(env_name="windows")
def is_image_2_brighter(image_path_1: str, image_path_2: str) -> bool:
    """
    Check if the second image is brighter than the first image.

    Args:
        image_path_1(str): The path to the first image.
        image_path_2(str): The path to the second image.
    """

    def brightness(image_path: str) -> float:
        # Load the image
        img = cv2.imread(image_path)
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return float(np.average(norm(img, axis=2)) / np.sqrt(3))
        else:
            # Grayscale
            return float(np.average(img))

    brightness_1 = brightness(image_path_1)
    brightness_2 = brightness(image_path_2)

    return brightness_2 > brightness_1


@evaluator(env_name="windows",local=True)
def is_img_url_in_clipboard() -> bool:
    """
    Check if the clipboard contains a valid URL or a Data URI that is specific to images.

    Args:
        env (Environment): The current testing environment, used to simulate clipboard functionality.

    Returns:
        bool: True if a valid URL or Data URI specific to images is found in the clipboard, False otherwise.
    """
    clipboard_content = pyperclip.paste()  # Simulate clipboard paste action
    data_uri_pattern = re.compile(
        r"^data:image/(png|jpeg|gif|svg\+xml|bmp|webp);base64,[A-Za-z0-9+/]+={0,2}$",
        re.IGNORECASE,
    )
    is_valid_image_data = bool(re.match(data_uri_pattern, clipboard_content))
    url_pattern = re.compile(
        r"^(https?://)"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    is_valid_url = bool(re.match(url_pattern, clipboard_content))
    if is_valid_url or is_valid_image_data:
        return True
    return False


@evaluator(env_name="windows")
def is_github_repo_url_in_clipboard(keyword: str) -> bool:
    """
    Check if the clipboard contains a valid GitHub repository URL.

    Returns:
        bool: True if the clipboard content is a valid GitHub repository URL, False otherwise.
    """
    clipboard_content = pyperclip.paste()  # Access the clipboard content
    if keyword.lower() not in clipboard_content:
        return False
    github_repo_pattern = re.compile(
        r"^https?://"  # Protocol
        r"github\.com/"  # Domain
        r"[^/]+/"  # Username
        r"[^/]+/?$",  # Repository name, optional trailing slash
        re.IGNORECASE,
    )
    return bool(re.match(github_repo_pattern, clipboard_content))
    # return is_github_repo_url(clipboard_content)


@evaluator(env_name="windows")
def is_software_installed(package_name: str) -> bool:
    """
    Check if the specified software is installed on a Windows system.

    Args:
        package_name (str): The name of the software or executable to check.

    Returns:
        bool: True if the software is installed, False otherwise.
    """
    try:
        # Use 'where' command to check if the executable is in the PATH
        subprocess.check_call(
            ["where", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        # If 'where' command fails, try checking with winget (if available)
        try:
            result = subprocess.run(
                ["winget", "list", "--id", package_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return package_name.lower() in result.stdout.lower()
        except FileNotFoundError:
            # winget is not available, try checking with chocolatey (if available)
            try:
                result = subprocess.run(
                    ["choco", "list", "--local-only", package_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return package_name.lower() in result.stdout.lower()
            except FileNotFoundError:
                # Neither winget nor chocolatey is available
                return False


@cache
def get_file_url_hash(url):
    response = requests.get(url)
    response.raise_for_status()
    return hashlib.sha256(response.content).hexdigest()


@evaluator(env_name="windows")
def download_and_verify_file(url: str, file_path: str) -> bool:
    # Check if the file was downloaded
    if not os.path.isfile(file_path):
        return False

    # Calculate the hash of the downloaded file
    with open(file_path, "rb") as f:
        file_data = f.read()
        downloaded_file_hash = hashlib.sha256(file_data).hexdigest()

    # Get the file content directly from the URL
    try:
        original_file_hash = get_file_url_hash(url)
    except requests.RequestException:
        return False

    # Compare the hashes
    return downloaded_file_hash == original_file_hash


@evaluator(env_name="windows")
def download_from_clipboard_and_verify_file(file_path: str) -> bool:
    # Check if the file was downloaded
    if not os.path.isfile(file_path):
        return False

    # Calculate the hash of the downloaded file
    with open(file_path, "rb") as f:
        file_data = f.read()
        downloaded_file_hash = hashlib.sha256(file_data).hexdigest()

    # Get the url from clipboard
    content = pyperclip.paste()
    """
    Problem:
        1. There exist infinite possibilities of the downloable format in the clipboard. Not sure if we need to verify the format.
    """
    # Get the file content directly from the URL
    try:
        original_file_hash = get_file_url_hash(content)
    except requests.RequestException:
        return False

    # Compare the hashes
    return downloaded_file_hash == original_file_hash


@evaluator(env_name="windows")
def check_color_scheme(assume: str) -> bool:
    # 使用 PowerShell 检查 Windows 的颜色方案
    command = [
        "powershell",
        "-Command",
        "Get-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize' -Name AppsUseLightTheme"
    ]

    try:
        # 执行 PowerShell 命令
        out = subprocess.check_output(command, text=True, shell=True)

        # 检查输出
        if "AppsUseLightTheme" in out:
            # AppsUseLightTheme = 1 表示浅色模式，0 表示深色模式
            if "1" in out and assume == "light":
                return True
            elif "0" in out and assume == "dark":
                return True
    except subprocess.CalledProcessError:
        pass

    return False

import pygetwindow as gw
@evaluator(env_name="windows",local=True)
def check_text_in_current_window_name(text: str) -> bool:
    print("check_text_in_current_window_name", text)
    try:
        # 获取当前活动窗口
        active_window = gw.getActiveWindow()
        # 获取窗口标题
        window_title = active_window.title

        # 检查文本是否在窗口标题中
        return text in window_title
    except Exception:
        return False


@evaluator(env_name="windows")
def check_current_window_process(assume: str) -> bool:
    try:
        # 获取当前活动窗口
        active_window = gw.getActiveWindow()

        # 获取窗口的进程 ID (PID)
        pid = active_window._hWnd  # 在 Windows 中，窗口句柄可以用于获取 PID

        # 通过 PID 获取进程信息
        process = psutil.Process(pid)

        # 检查进程名称是否匹配
        return assume.strip().lower() == process.name().lower()
    except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            AttributeError,
            Exception
    ):
        return False


@evaluator(env_name="windows")
def check_file_exist(file_path: str) -> bool:
    # 规范化路径（将斜杠转换为反斜杠）
    normalized_path = os.path.normpath(file_path)

    # 检查文件是否存在
    return os.path.isfile(normalized_path)


@evaluator(env_name="windows")
def check_file_content(file_path: str, content: str) -> bool:
    # 规范化路径（将斜杠转换为反斜杠）
    normalized_path = os.path.normpath(file_path)

    # 检查文件是否存在
    if not os.path.isfile(normalized_path):
        return False

    # 读取文件内容并检查是否包含指定内容
    with open(normalized_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    return content in file_content


@evaluator(env_name="windows")
def empty_evaluator() -> bool:
    return False


@evaluator(env_name="windows")
def is_process_open(process_name: str) -> bool:
    """
    Check if the given process is currently running.

    Args:
        process_name(str): The process name to check.
    """
    # 确保进程名称包含 .exe 后缀（如果未提供）
    if not process_name.lower().endswith(".exe"):
        process_name += ".exe"

    # 遍历所有进程
    for process in psutil.process_iter(["name"]):
        try:
            if process_name.lower() == process.info["name"].lower():  # type: ignore
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


@evaluator(env_name="windows")
def check_app_usage_history(app_name: str) -> bool:
    """
    Check if the given application has been in the usage history.

    Args:
        app_name(str): The name of the application to check.

    Returns:
        bool: True if the app was recently used, False otherwise.
    """
    # 确保应用程序名称包含 .exe 后缀（如果未提供）
    if not app_name.lower().endswith(".exe"):
        app_name += ".exe"

    # 遍历所有进程
    for process in psutil.process_iter(["name", "create_time"]):
        try:
            # 检查进程名称是否匹配
            if app_name.lower() == process.info["name"].lower():
                # 检查进程是否在最近一小时内启动
                if time.time() - process.info["create_time"] < 3600:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


@evaluator(env_name="windows")
def check_process_closed(app_name: str) -> bool:
    """
    Verify that the specified process is not running.

    Args:
        app_name(str): The application name to check for its absence.

    Returns:
        bool: True if the app is not running, False otherwise.
    """
    # 确保应用程序名称包含 .exe 后缀（如果未提供）
    if not app_name.lower().endswith(".exe"):
        app_name += ".exe"

    # 遍历所有进程
    for proc in psutil.process_iter(["name"]):
        try:
            # 检查进程名称是否匹配
            if app_name.lower() == proc.info["name"].lower():
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return True

#
# import hashlib
# #import winreg
#
def get_windows_wallpaper_path() -> str:
    """
    Get the current desktop wallpaper path from the Windows registry.
    """
    try:
        # 打开注册表键
        reg_key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Control Panel\Desktop"
        )
        # 读取 WallPaper 值
        wallpaper_path, _ = winreg.QueryValueEx(reg_key, "WallPaper")
        winreg.CloseKey(reg_key)
        return wallpaper_path
    except Exception:
        return ""

@evaluator(env_name="windows")
def verify_background(photo_path: str) -> bool:
    """
    Verify that the specified photo is currently set as the desktop background.

    Args:
        photo_path (str): The path to the photo file.

    Returns:
        bool: True if the photo is the current background, False otherwise.
    """
    # 获取当前桌面背景的路径
    current_background = get_windows_wallpaper_path()
    if not current_background:
        return False

    # 比较文件路径
    if os.path.normpath(photo_path) == os.path.normpath(current_background):
        return True

    # 如果路径不匹配，比较文件内容（通过哈希值）
    if os.path.exists(photo_path) and os.path.exists(current_background):
        with open(photo_path, "rb") as f:
            original_hash = hashlib.sha256(f.read()).hexdigest()
        with open(current_background, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        return original_hash == current_hash

    return False

import pyperclip

@evaluator(env_name="windows")
def is_torch_matmul_example_copied_correctly() -> bool:
    """
    Verify if the clipboard contains the correct torch.matmul example snippets from PyTorch 1.13 documentation.
    """

    def contains_required_strings(clipboard_content: str, required_strings: list) -> bool:
        """
        Check if all required strings are present in the clipboard content.

        Args:
            clipboard_content (str): The content from the clipboard.
            required_strings (list): A list of required strings to check.

        Returns:
            bool: True if all required strings are found in the clipboard content, False otherwise.
        """
        for string in required_strings:
            if string not in clipboard_content:
                return False
        return True

    # 定义需要检查的代码片段
    required_strings = [
        "tensor1 = torch.randn",
        "tensor2 = torch.randn",
        "torch.matmul(tensor1, tensor2).size()",
    ]

    # 获取剪贴板内容
    clipboard_content = pyperclip.paste().strip()
    if not clipboard_content:
        return False

    # 检查是否包含所有必需的代码片段
    return contains_required_strings(clipboard_content, required_strings)

@evaluator(env_name="windows")
def check_directory_exists(dir_path: str) -> bool:
    """Check if the specified directory exists."""
    # 规范化路径（将斜杠转换为反斜杠）
    normalized_path = os.path.normpath(dir_path)

    # 检查目录是否存在
    return os.path.isdir(normalized_path)


@evaluator(env_name="windows")
def verify_files_copied(source_dir: str, target_dir: str, file_extension: str) -> bool:
    """Verify that files were copied correctly."""
    # 规范化路径（将斜杠转换为反斜杠）
    source_dir = os.path.normpath(source_dir)
    target_dir = os.path.normpath(target_dir)

    # 获取源目录和目标目录中指定扩展名的文件
    source_files = {
        file for file in os.listdir(source_dir) if file.endswith(f".{file_extension}")
    }
    target_files = {
        file for file in os.listdir(target_dir) if file.endswith(f".{file_extension}")
    }

    # 比较两个集合
    return source_files == target_files


@evaluator(env_name="windows", local=True)
def check_contain_input_text_list(texts: list[str], env) -> bool:
    """
    Check if all provided search terms were entered in the browser.

    Args:
        texts: A list of strings, each representing a search term that needs to be verified.
        env: The current testing environment, used to simulate browser interactions.

    Returns:
        bool: True if all search terms are found in the written text, False otherwise.
    """
    if env.trajectory:
        # 提取所有 "write_text" 操作中的输入文本
        inputs = [
            params["text"].lower()
            for action_name, params, _ in env.trajectory
            if action_name == "write_text"
        ]
        # 检查所有搜索词是否在输入文本中
        return all(
            any(term.lower() in input_text for input_text in inputs) for term in texts
        )
    return False


@evaluator(env_name="windows")
def is_google_maps_url_in_clipboard() -> bool:
    """
    Check if the clipboard contains a valid shortened Google Maps URL.
    """
    # 获取剪贴板内容
    clipboard_content = pyperclip.paste().strip()

    # 定义 Google Maps 短链接的正则表达式
    maps_url_pattern = re.compile(
        r"^https://maps\.app\.goo\.gl/[A-Za-z0-9]+$",
        re.IGNORECASE,
    )

    # 检查剪贴板内容是否匹配正则表达式
    return bool(re.match(maps_url_pattern, clipboard_content))


@evaluator(env_name="windows", local=True)
def check_contain_input_text(text: str, env) -> bool:
    print("check_contain_input_text", text)
    """
    Check if the input text is contained in the written text action in a case-insensitive manner.

    Args:
        text (str): The text to check for.
        env: The current testing environment, used to access the trajectory.

    Returns:
        bool: True if the input text is found in the written text action, False otherwise.
    """
    if env.trajectory:
        # 提取所有 "write_text" 操作中的输入文本
        inputs = [
            params["text"].lower()
            for action_name, params, _ in env.trajectory
            if action_name == "write_text"
        ]
        # 检查输入文本是否在任一输入中出现过
        return any(text.lower() in input_text for input_text in inputs)
    return False


import re
import requests
from bs4 import BeautifulSoup
from pyexcel_ods import get_data
#####################################
#########新增评估器####################


#####################################

windows_subtasks = [
    SubTask(
        id="ask_deepseek_details_huashi",
        description='open Google Chrome and navigate to"{url}" then login in,and 进入 "DeepSeek",询问 "{content}"',
        attribute_dict={"url": "url", "content": "content"},
        output_type="content",
        evaluator_generator=lambda url, content: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("一网通办 - Google Chrome"),
                check_text_in_current_window_name("华师AI智聊 - Google Chrome"),
                check_contain_input_text(content)

            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="check_message_in_yizhanshi",
        description='打开Google Chrome 并且导航到"{url}" 并且"login in",然后进入"消息中心"中查看消息',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("一网通办 - Google Chrome"),
                check_text_in_current_window_name("收件箱 - Google Chrome"),

            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="check_task_in_yizhanshi",
        description='打开Google Chrome 并且导航到"{url}" (一站式服务平台）并且"login in",在"任务中心"中查看待办任务',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("一网通办 - Google Chrome"),
                check_text_in_current_window_name("任务列表 - Google Chrome"),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(  # test:66.66%
        id="check_mail_in_yizhanshi",
        description='在Google Chrome 导航到"{url}"(一站式服务平台）",然后在"我的邮件"中查看邮件',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("一网通办 - Google Chrome"),
                check_text_in_current_window_name("腾讯企业邮箱 - Google Chrome"),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="check_book_in_yizhanshi",
        description='打开Google Chrome 并且导航到"{url}" 并且"login in",然后进入图书馆主页',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("一网通办 - Google Chrome"),
                check_text_in_current_window_name("首页 - Google Chrome")

            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="check_ccnuhomepage_in_yizhanshi",
        description='打开Google Chrome 并且导航到"{url}" 并且"login in",然后进入华师主页',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("一网通办 - Google Chrome"),
                check_text_in_current_window_name("华中师范大学 - Google Chrome")

            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="check_datastructure_in_mooc",
        description='打开Google Chrome 并且导航到"{url}" ,然后搜索"数据结构"课程',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("中国大学MOOC_优质在线课程学习平台 - Google Chrome"),
                check_text_in_current_window_name("搜索课程_中国大学MOOC(慕课) - Google Chrome"),

            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="check_cloud_in_mooc",
        description='打开Google Chrome 并且导航到"{url}",然后点击"我的学校云"，查看华师的相关课程',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [
                check_text_in_current_window_name("New Tab - Google Chrome"),
                check_text_in_current_window_name("中国大学MOOC_优质在线课程学习平台 - Google Chrome"),
                check_text_in_current_window_name("华中师范大学_中国大学MOOC(慕课) - Google Chrome"),

            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="check_makesi_in_mooc",
        description='打开Google Chrome 并且导航到"{url}",然后进入"马克思主义基本原理"课程',
        attribute_dict={"url": "url"},
        output_type="None",
        evaluator_generator=lambda url: path_graph(
            [

                check_text_in_current_window_name("New Tab - Google Chrome"),

                check_text_in_current_window_name("中国大学MOOC_优质在线课程学习平台 - Google Chrome"),

                check_text_in_current_window_name("马克思主义基本原理_中国大学MOOC(慕课) - Google Chrome"),

            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
            id="microsoft_search_image",
            description='Use Microsoft to search for an image using the keyword "{keyword}" and copy the URL of the image to the clipboard.',
            attribute_dict={"keyword": "keyword"},
            output_type="None",
            evaluator_generator=lambda keyword: path_graph(
                [
                    check_text_in_current_window_name("Chrome"),
                    check_contain_input_text(keyword),
                    is_img_url_in_clipboard(),
                ],
                create_using=DiGraph,
            ),
        ),
    SubTask(
            id="chrome_xiaoya_course",
            description="使用Chrome浏览器打开华师小雅中对应课程",
            attribute_dict={"course_name":"course_name"},
            output_type="None",
            evaluator_generator=lambda course_name: path_graph(
                [
                    check_text_in_current_window_name("Chrome"),
                    check_contain_input_text("https://ccnu.ai-augmented.com/app/jx-web/mycourse"),
                    check_text_in_current_window_name("小雅"),
                    check_text_in_current_window_name("课程内容")
                ],
                create_using=DiGraph,
            ),
        ),
    SubTask(
        id="chrome_xiaoya_tasks",
        description="使用Chrome浏览器打开华师小雅中查看全部任务",
        attribute_dict={},
        output_type="None",
        evaluator_generator=lambda : path_graph(
            [
                check_text_in_current_window_name("Chrome"),
                check_contain_input_text("https://ccnu.ai-augmented.com/app/jx-web/mycourse"),
                check_text_in_current_window_name("小雅"),
                check_text_in_current_window_name("课程内容")
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="chrome_xiaoya_write_document",
        description="使用Chrome浏览器打开华师小雅中的我的文档，并写入标题为title相关内容",
        attribute_dict={"title":"title"},
        output_type="None",
        evaluator_generator=lambda title: path_graph(
            [
                check_text_in_current_window_name("Chrome"),
                check_contain_input_text("https://ccnu.ai-augmented.com/app/jx-web/mycourse"),
                check_text_in_current_window_name("小雅"),
                check_text_in_current_window_name("我的文档"),
                check_contain_input_text(title)
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="chrome_xiaoya_check_document",
        description="使用Chrome浏览器打开华师小雅中的我的文档，并查看某个笔记的内容",
        attribute_dict={"title":"title"},
        output_type="Content",
        evaluator_generator=lambda title: path_graph(
            [
                check_text_in_current_window_name("Chrome"),
                check_contain_input_text("https://ccnu.ai-augmented.com/app/jx-web/mynote/"),
                check_text_in_current_window_name("我的文档"),
                check_text_in_current_window_name(title)
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="chrome_xiaoya_ai_ask",
        description="使用Chrome浏览器打开华师小雅中的AI助手，并询问问题",
        attribute_dict={"content":"content"},
        output_type="Content",
        evaluator_generator=lambda content: path_graph(
            [
                check_text_in_current_window_name("Chrome"),
                check_contain_input_text("https://ccnu.ai-augmented.com/app/jx-web/mycourse/"),
                check_text_in_current_window_name("小雅"),
                check_text_in_current_window_name("课程内容"),
                check_contain_input_text(content)
            ],
            create_using=DiGraph,
        ),
    ),
]


if __name__ == "__main__":
    generator = TaskGenerator(attribute_pool={})
