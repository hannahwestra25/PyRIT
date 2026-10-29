# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import hashlib
import math
import string
from collections import Counter
from io import BytesIO
from pathlib import Path
from threading import get_ident
from unittest.mock import patch

import aiofiles
import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFont

from pyrit.converter import UglyGerryConverter, get_converter_modalities
from pyrit.registry import ConverterRegistry


async def _read_image_async(path: str) -> Image.Image:
    async with aiofiles.open(path, "rb") as file:
        data = await file.read()
    with Image.open(BytesIO(data)) as image:
        assert image.format == "PNG"
        return image.convert("RGB")


def test_bundled_font_is_unmodified() -> None:
    data = UglyGerryConverter._FONT_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == "38332905de7a8bcbda48038edd7cc5dfd26924b3991eebfc9f8ca84b0b35f1e4"
    font = ImageFont.truetype(BytesIO(data), size=64)
    assert font.getname() == ("Gerry", "Regular")
    for character in string.ascii_uppercase + "!',.?":
        assert font.getmask(character).getbbox() is not None


@pytest.mark.usefixtures("patch_central_database")
async def test_convert_async_renders_the_bundled_district_glyphs() -> None:
    converter = UglyGerryConverter()
    result = await converter.convert_async(prompt="Hello, World!")
    with await _read_image_async(result.output_text) as actual:
        font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
        left, top, right, bottom = font.getbbox("HELLO, WORLD!")
        with Image.new("RGB", (right - left + 40, bottom - top + 40), "white") as expected:
            ImageDraw.Draw(expected).text((20 - left, 20 - top), "HELLO, WORLD!", fill="black", font=font)
            assert actual.size == expected.size
            assert actual.tobytes() == expected.tobytes()
        assert result.output_type == "image_path"
        assert ImageChops.difference(actual, Image.new("RGB", actual.size, "white")).getbbox() is not None


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("concurrent", [False, True])
async def test_convert_async_uses_unique_files(concurrent: bool) -> None:
    converter = UglyGerryConverter()
    if concurrent:
        results = await asyncio.gather(*(converter.convert_async(prompt="Same prompt") for _ in range(3)))
    else:
        results = [await converter.convert_async(prompt="Same prompt") for _ in range(3)]
    assert len({result.output_text for result in results}) == 3
    images = [await _read_image_async(result.output_text) for result in results]
    try:
        assert all(image.tobytes() == images[0].tobytes() for image in images)
    finally:
        for image in images:
            image.close()


@pytest.mark.usefixtures("patch_central_database")
async def test_convert_async_normalizes_case_and_line_endings() -> None:
    converter = UglyGerryConverter()
    results = [
        await converter.convert_async(prompt=prompt)
        for prompt in ("hello\n\nworld!", "HELLO\r\n\r\nWORLD!", "Hello\r\rWorld!")
    ]
    images = [await _read_image_async(result.output_text) for result in results]
    try:
        assert all(image.tobytes() == images[0].tobytes() for image in images)
    finally:
        for image in images:
            image.close()


@pytest.mark.parametrize("prompt", ["", " ", "\n\r"])
async def test_convert_async_rejects_blank_prompt(prompt: str) -> None:
    with pytest.raises(ValueError, match="valid text"):
        await UglyGerryConverter().convert_async(prompt=prompt)


@pytest.mark.parametrize("character", ["0", "9", ":", "-", "\t", "\u00e9", "\u00df", "\U0001f600", "\x00"])
async def test_convert_async_rejects_unsupported_characters(character: str) -> None:
    with pytest.raises(ValueError, match="does not support"):
        await UglyGerryConverter().convert_async(prompt=f"Hello{character}")


async def test_convert_async_rejects_non_text_input() -> None:
    with pytest.raises(ValueError, match="Input type image_path not supported"):
        await UglyGerryConverter().convert_async(prompt="test.png", input_type="image_path")


@pytest.mark.parametrize(
    "parameters",
    [
        {"font_size": 0},
        {"font_size": -1},
        {"font_size": 1.5},
        {"font_size": True},
        {"max_width": 0},
        {"max_width": 40, "padding": 20},
        {"padding": -1},
        {"color": (256, 0, 0)},
        {"background_color": (0, 0)},
        {"color": (255, 255, 255)},
    ],
)
def test_init_rejects_invalid_parameters(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UglyGerryConverter(**parameters)  # type: ignore[arg-type]


def test_init_does_not_fall_back_when_font_is_missing(tmp_path: Path) -> None:
    with patch.object(UglyGerryConverter, "_FONT_PATH", tmp_path / "missing.otf"):
        with pytest.raises(FileNotFoundError):
            UglyGerryConverter()


def test_init_does_not_fall_back_when_font_is_corrupt(tmp_path: Path) -> None:
    font_path = tmp_path / "corrupt.otf"
    font_path.write_bytes(b"not a font")
    with patch.object(UglyGerryConverter, "_FONT_PATH", font_path):
        with pytest.raises(OSError):
            UglyGerryConverter()


@pytest.mark.parametrize(
    "text",
    ["W" * 50, "I" * 100, string.ascii_uppercase, "HI THERE\n\nWORLD!", "VV\nAA\nYY\nGG"],
)
def test_wrapping_preserves_characters_and_fits_pixel_width(text: str) -> None:
    converter = UglyGerryConverter(max_width=180)
    font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
    lines = [line for line, _ in converter._iter_wrapped_lines(text=text, font=font)]
    assert "".join(lines) == text.replace("\n", "")
    assert all(font.getbbox(line)[2] - font.getbbox(line)[0] <= 140 for line in lines)
    with Image.open(BytesIO(converter._render_png(text))) as image:
        assert image.width <= 180
        assert image.height > 64
        with ImageChops.difference(image, Image.new("RGB", image.size, "white")) as ink:
            bbox = ink.getbbox()
            assert bbox is not None
            assert bbox[0] >= 20 and bbox[1] >= 20
            assert bbox[2] <= image.width - 20 and bbox[3] <= image.height - 20


def test_wrapping_measures_repeated_candidates_once() -> None:
    converter = UglyGerryConverter()
    font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
    prompt = "A" * 10_000
    with patch.object(font, "getbbox", wraps=font.getbbox) as measure:
        wrapped = list(converter._iter_wrapped_lines(text=prompt, font=font))
    lines = [line for line, _ in wrapped]
    assert "".join(lines) == prompt
    assert lines[:-1] == ["A" * 14] * (len(lines) - 1)
    assert lines[-1] == "A" * 4
    counts = Counter(call.args[0] for call in measure.call_args_list)
    assert len(counts) == 15
    assert all(count == 1 for count in counts.values())


def test_wrapping_measurement_cache_is_bounded() -> None:
    converter = UglyGerryConverter(max_width=122)
    font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
    with (
        patch.object(converter, "_MEASUREMENT_CACHE_SIZE", 2),
        patch.object(font, "getbbox", wraps=font.getbbox) as measure,
    ):
        lines = [line for line, _ in converter._iter_wrapped_lines(text="ABAB", font=font)]
    assert lines == ["A", "B", "A", "B"]
    counts = Counter(call.args[0] for call in measure.call_args_list)
    assert counts["A"] == 2
    assert counts["B"] == 2


def test_wrapping_measurement_cache_is_local_to_each_layout() -> None:
    converter = UglyGerryConverter()
    font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
    with patch.object(font, "getbbox", wraps=font.getbbox) as measure:
        first = list(converter._iter_wrapped_lines(text="AAA", font=font))
        second = list(converter._iter_wrapped_lines(text="AAA", font=font))
    assert first == second
    counts = Counter(call.args[0] for call in measure.call_args_list)
    assert counts == {"A": 2, "AA": 2, "AAA": 2}


def test_rendering_reuses_measurements_for_completed_lines() -> None:
    converter = UglyGerryConverter()
    font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
    with (
        patch.object(ImageFont, "truetype", return_value=font),
        patch.object(font, "getbbox", wraps=font.getbbox) as measure,
    ):
        converter._render_png("A" * 70)
    counts = Counter(call.args[0] for call in measure.call_args_list)
    assert len(counts) == 15
    assert all(count == 1 for count in counts.values())


def test_rendering_rejects_oversized_layout_before_consuming_all_lines() -> None:
    converter = UglyGerryConverter()
    font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
    prompt = "A" * 10_000
    layout = converter._iter_wrapped_lines(text=prompt, font=font)
    with (
        patch.object(converter, "_iter_wrapped_lines", return_value=layout),
        patch.object(ImageFont, "truetype", return_value=font),
        patch.object(Image, "new", side_effect=AssertionError("Oversized images must not be allocated")),
    ):
        with pytest.raises(ValueError, match="exceeds 16 million pixels"):
            converter._render_png(prompt)
    remaining = list(layout)
    total_lines = math.ceil(len(prompt) / 14)
    assert 0 < len(remaining) < total_lines
    completed = total_lines - len(remaining)
    left, top, right, bottom = font.getbbox("A" * 14)
    width = right - left + 40
    line_height = bottom - top + 16
    assert width * (bottom - top + (completed - 2) * line_height + 40) <= converter._MAX_PIXELS
    assert width * (bottom - top + (completed - 1) * line_height + 40) > converter._MAX_PIXELS


@pytest.mark.parametrize("line_count", [1, 3])
@pytest.mark.parametrize("padding", [0, 20])
def test_rendering_preserves_the_exact_pixel_budget_boundary(line_count: int, padding: int) -> None:
    converter = UglyGerryConverter(padding=padding)
    font = ImageFont.truetype(str(converter._FONT_PATH), size=64)
    left, top, right, bottom = font.getbbox("A")
    width = right - left + 2 * padding
    height = bottom - top + (line_count - 1) * (bottom - top + 16) + 2 * padding
    prompt = "\n".join(["A"] * line_count)
    with patch.object(converter, "_MAX_PIXELS", width * height):
        with Image.open(BytesIO(converter._render_png(prompt))) as image:
            assert image.size == (width, height)
    with patch.object(converter, "_MAX_PIXELS", width * height - 1):
        with pytest.raises(ValueError, match="exceeds 16 million pixels"):
            converter._render_png(prompt)


async def test_convert_async_rejects_width_smaller_than_one_glyph() -> None:
    with pytest.raises(ValueError, match="too small to render"):
        await UglyGerryConverter(max_width=41).convert_async(prompt="W")


def test_rendering_keeps_custom_colors() -> None:
    converter = UglyGerryConverter(color=(10, 20, 30), background_color=(240, 230, 220), padding=5)
    with Image.open(BytesIO(converter._render_png("HELLO"))) as image:
        assert image.getpixel((0, 0)) == (240, 230, 220)
        assert (10, 20, 30) in image.get_flattened_data()


async def test_convert_async_rejects_oversized_images_before_saving() -> None:
    converter = UglyGerryConverter()
    with patch.object(converter, "_MAX_PIXELS", 1):
        with pytest.raises(ValueError, match="exceeds 16 million pixels"):
            await converter.convert_async(prompt="Hello")


@pytest.mark.usefixtures("patch_central_database")
async def test_convert_async_renders_off_the_event_loop() -> None:
    converter = UglyGerryConverter()
    render = converter._render_png
    event_loop_thread = get_ident()
    worker_threads: list[int] = []

    def _render(text: str) -> bytes:
        worker_threads.append(get_ident())
        return render(text)

    with patch.object(converter, "_render_png", side_effect=_render):
        await converter.convert_async(prompt="Hello")
    assert worker_threads and all(thread != event_loop_thread for thread in worker_threads)


@pytest.mark.usefixtures("patch_central_database")
async def test_convert_async_propagates_storage_errors() -> None:
    converter = UglyGerryConverter()
    with patch("pyrit.converter.ugly_gerry_converter.data_serializer_factory", side_effect=OSError("storage failed")):
        with pytest.raises(OSError, match="storage failed"):
            await converter.convert_async(prompt="Hello")


@pytest.mark.parametrize(
    "parameters",
    [
        {"font_size": 32},
        {"max_width": 600},
        {"padding": 0},
        {"color": (255, 0, 0)},
        {"background_color": (240, 240, 240)},
    ],
)
def test_identifier_includes_rendering_parameters(parameters: dict[str, object]) -> None:
    default = UglyGerryConverter()
    configured = UglyGerryConverter(**parameters)  # type: ignore[arg-type]
    assert configured.get_identifier().hash != default.get_identifier().hash
    for name, value in parameters.items():
        expected = list(value) if isinstance(value, tuple) else value
        assert configured.get_identifier().params[name] == expected


def test_converter_is_discoverable_and_buildable() -> None:
    registry = ConverterRegistry()
    converter = registry.create_instance("UglyGerryConverter", font_size=32)
    assert isinstance(converter, UglyGerryConverter)
    assert converter.get_identifier().params["font_size"] == 32
    assert ("UglyGerryConverter", ["text"], ["image_path"]) in get_converter_modalities()
