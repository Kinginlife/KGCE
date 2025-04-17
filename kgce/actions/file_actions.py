import base64
import os
from io import BytesIO

from PIL import Image

from kgce.core import action


@action
def save_base64_image(image: str, path: str = "image.png") -> None:
    """
    Save a Base64-encoded image to a file.

    Args:
        image: The Base64-encoded image string.
        path: The file path to save the image. Defaults to "image.png".
    """
    # Decode the Base64 image
    image_data = base64.b64decode(image)
    image = Image.open(BytesIO(image_data))

    # Ensure the path is platform-agnostic
    path = os.path.normpath(path)

    # Save the image
    image.save(path)