# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import math
import string
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from pyrit.converter.base_image_text_converter import _BaseImageTextConverter
from pyrit.converter.converter import ConverterResult
from pyrit.memory import data_serializer_factory
from pyrit.models import ComponentIdentifier, PromptDataType


class UglyGerryConverter(_BaseImageTextConverter):
    """
    Render text as a PNG using congressional-district shapes from the bundled Ugly Gerry font.

    ASCII letters are rendered as capitals. Spaces, newlines, and the punctuation
    ``! ' , . ?`` are supported; other characters are rejected rather than lost.
    Lines wrap at character boundaries to fit the configured maximum image width.
    No font installation, network access, or model is required.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("image_path",)
    _FONT_PATH = Path(__file__).parent / "fonts" / "Gerry.otf"
    _SUPPORTED_CHARACTERS = frozenset(string.ascii_letters + " \n\r!',.?")
    _MAX_PIXELS = 16_000_000

    def __init__(
        self,
        *,
        font_size: int = 64,
        max_width: int = 1200,
        padding: int = 20,
        color: tuple[int, int, int] = (0, 0, 0),
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """
        Initialize the local font renderer.

        Args:
            font_size (int): Font size in pixels. Defaults to 64.
            max_width (int): Maximum image width, including padding. Defaults to 1200.
            padding (int): Blank border in pixels. Defaults to 20.
            color (tuple[int, int, int]): Text RGB color. Defaults to black.
            background_color (tuple[int, int, int]): Background RGB color. Defaults to white.

        Raises:
            ValueError: If sizes or colors are invalid, or the colors are identical.
            OSError: If the bundled font is missing or cannot be loaded.
        """
        super().__init__()
        for name, value, minimum in (("font_size", font_size, 1), ("max_width", max_width, 1), ("padding", padding, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
        if max_width <= 2 * padding:
            raise ValueError("max_width must be greater than twice padding")
        self._validate_color(color)
        self._validate_color(background_color)
        if color == background_color:
            raise ValueError("color and background_color must differ so the text is visible")
        self._font_size = font_size
        self._max_width = max_width
        self._padding = padding
        self._color = color
        self._background_color = background_color
        self._font_data = self._FONT_PATH.read_bytes()
        ImageFont.truetype(BytesIO(self._font_data), size=font_size)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Render a prompt and save a uniquely named PNG through PyRIT's image serializer.

        Args:
            prompt (str): Text containing only the supported letters, punctuation, and whitespace.
            input_type (PromptDataType): Must be ``text``.

        Returns:
            ConverterResult: The generated PNG path with output type ``image_path``.

        Raises:
            ValueError: If the input type or characters are unsupported, the prompt is blank,
                a character cannot fit the width, or the image would exceed 16 million pixels.
        """
        if not self.input_supported(input_type):
            raise ValueError(f"Input type {input_type} not supported")
        if not prompt.strip():
            raise ValueError("Please provide valid text value")
        unsupported = set(prompt) - self._SUPPORTED_CHARACTERS
        if unsupported:
            raise ValueError(f"Ugly Gerry does not support these characters: {sorted(unsupported)!r}")
        text = prompt.replace("\r\n", "\n").replace("\r", "\n").upper()
        image_bytes = await asyncio.to_thread(self._render_png, text)
        serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path", extension="png")
        await serializer.save_data_async(data=image_bytes)
        return ConverterResult(output_text=str(serializer.value), output_type="image_path")

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "font_size": self._font_size,
                "max_width": self._max_width,
                "padding": self._padding,
                "color": self._color,
                "background_color": self._background_color,
            }
        )

    def _wrap_lines(self, *, text: str, font: FreeTypeFont) -> list[str]:
        lines: list[str] = []
        width = self._max_width - 2 * self._padding
        for paragraph in text.split("\n"):
            line = ""
            for character in paragraph:
                candidate = line + character
                left, _, right, _ = font.getbbox(candidate)
                if right - left > width:
                    if not line:
                        raise ValueError(f"max_width is too small to render character {character!r}")
                    lines.append(line)
                    line = character
                    left, _, right, _ = font.getbbox(line)
                    if right - left > width:
                        raise ValueError(f"max_width is too small to render character {character!r}")
                else:
                    line = candidate
            lines.append(line)
        return lines

    def _render_png(self, text: str) -> bytes:
        font = ImageFont.truetype(BytesIO(self._font_data), size=self._font_size)
        lines = self._wrap_lines(text=text, font=font)
        bounds = [font.getbbox(line) for line in lines]
        top = min(box[1] for box in bounds)
        bottom = max(box[3] for box in bounds)
        spacing = self._font_size // 4
        line_height = max(1, bottom - top) + spacing
        width = max(1, math.ceil(max(box[2] - box[0] for box in bounds))) + 2 * self._padding
        height = math.ceil(max(1, bottom - top) + (len(lines) - 1) * line_height) + 2 * self._padding
        if width * height > self._MAX_PIXELS:
            raise ValueError("Rendered image exceeds 16 million pixels; shorten the prompt or reduce font_size")
        with Image.new("RGB", (width, height), self._background_color) as image, BytesIO() as output:
            draw = ImageDraw.Draw(image)
            for index, (line, box) in enumerate(zip(lines, bounds, strict=True)):
                draw.text(
                    (self._padding - box[0], self._padding - top + index * line_height),
                    line,
                    font=font,
                    fill=self._color,
                )
            image.save(output, format="PNG")
            return output.getvalue()
