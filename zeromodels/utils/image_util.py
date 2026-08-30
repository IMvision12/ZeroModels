from __future__ import annotations

import io
import os
from typing import Optional, Sequence, Tuple, Union

import keras
import numpy as np
import PIL.Image
from PIL import Image

ImageInput = Union[str, bytes, bytearray, np.ndarray, "PIL.Image.Image"]
BatchImageInput = Union[ImageInput, Sequence[ImageInput]]


def get_data_format(data_format: Optional[str] = None) -> str:
    """Return a concrete data format string.

    Args:
        data_format: Either ``"channels_first"``, ``"channels_last"``, or
            ``None``. When ``None``, defaults to the global Keras setting from
            ``keras.config.image_data_format()``.
    """
    if data_format is None:
        return keras.config.image_data_format()
    if data_format not in ("channels_first", "channels_last"):
        raise ValueError(
            "data_format must be 'channels_first', 'channels_last', or None; "
            f"got {data_format!r}."
        )
    return data_format


def standardize_input_shape(
    image_size: Union[int, Tuple[int, ...]],
    data_format: Optional[str] = None,
) -> Tuple[int, int, int]:
    """Normalize a flexible ``image_size`` into a canonical 3-tuple.

    Accepts:

    * ``int N``: square image, builds ``(N, N, 3)`` for ``channels_last``
      or ``(3, N, N)`` for ``channels_first``.
    * ``(H, W)``: rectangular image, adds a 3-channel dim per data format.
    * ``(H, W, C)`` or ``(C, H, W)``: already a 3-tuple. The channel
      dimension (``C in {1, 3, 4}``) must sit in the position required by
      the active data format; mismatches raise ``ValueError``.

    Args:
        image_size: Flexible spec, int, 2-tuple, or 3-tuple.
        data_format: ``"channels_first"`` / ``"channels_last"`` / ``None``.
            ``None`` defaults to ``keras.config.image_data_format()``.

    Returns:
        A length-3 tuple ordered to match the resolved ``data_format``.
    """
    data_format = get_data_format(data_format)

    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError(f"image_size int must be positive, got {image_size}.")
        if data_format == "channels_last":
            return (image_size, image_size, 3)
        return (3, image_size, image_size)

    if not isinstance(image_size, (tuple, list)):
        raise TypeError(
            "image_size must be int, 2-tuple, or 3-tuple; got "
            f"{type(image_size).__name__}."
        )

    shape = tuple(int(d) for d in image_size)

    if any(d <= 0 for d in shape):
        raise ValueError(f"image_size dimensions must be positive, got {shape}.")

    if len(shape) == 2:
        h, w = shape
        if data_format == "channels_last":
            return (h, w, 3)
        return (3, h, w)

    if len(shape) == 3:
        if data_format == "channels_last":
            if shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"image_size {shape} does not match data_format "
                    "'channels_last'. Expected (H, W, C) with C in {1, 3, 4}; "
                    "for channels_first inputs call "
                    "keras.config.set_image_data_format('channels_first')."
                )
            return shape
        if shape[0] not in (1, 3, 4):
            raise ValueError(
                f"image_size {shape} does not match data_format "
                "'channels_first'. Expected (C, H, W) with C in {1, 3, 4}; "
                "for channels_last inputs call "
                "keras.config.set_image_data_format('channels_last')."
            )
        return shape

    raise ValueError(
        "image_size must be int, 2-tuple, or 3-tuple; got tuple of "
        f"length {len(shape)}."
    )


def load_image(image: ImageInput) -> np.ndarray:
    """Load an image from common sources into an ``(H, W, 3)`` uint8 RGB array.

    Accepted inputs:
        * ``str``: a local file path or an ``http(s)://`` URL.
        * ``bytes`` / ``bytearray``: raw encoded image bytes.
        * ``PIL.Image.Image``: returned as a copy converted to RGB.
        * ``np.ndarray``: assumed to already be an HWC RGB image. 2D arrays
          are broadcast across 3 channels; 4-channel arrays are truncated to
          RGB; float arrays in [0, 1] are scaled to uint8.
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape {arr.shape}.")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got shape {arr.shape}.")
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    if isinstance(image, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))

    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(image) as response:
                data = response.read()
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        return np.asarray(Image.open(image).convert("RGB"))

    raise TypeError(f"Unsupported image input type: {type(image).__name__}.")
