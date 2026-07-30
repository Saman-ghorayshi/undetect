"""Tests for undetect module."""
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from undetect import (
    load_image,
    strip_metadata,
    add_film_grain,
    jpeg_roundtrip,
    resize_cycle,
    dct_band_perturb,
    color_jitter,
    process_image,
    process_file,
    batch_process,
    analyze,
    STRENGTH_PROFILES,
)

def make_test_image(size=(100, 100), mode="RGB"):
    """Make a deterministic test image with some structure."""
    arr = np.zeros((size[1], size[0], 3) if mode == "RGB" else (size[1], size[0]), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            val = int(128 + 50 * np.sin(x * 0.1) * np.cos(y * 0.1))
            if mode == "RGB":
                arr[y, x] = [val, val // 2, 255 - val]
            else:
                arr[y, x] = val
    return Image.fromarray(arr, mode)


def make_image_file(path, size=(50, 50), format="JPEG"):
    img = make_test_image(size)
    img.save(path, format=format)
    return path


# --- metadata strip tests ---

def test_strip_metadata_returns_same_pixels():
    img = make_test_image((30, 30))
    # add fake metadata
    img.info["exif"] = b"fake_exif_data"
    clean = strip_metadata(img)
    assert np.array_equal(np.array(img), np.array(clean))

def test_strip_metadata_drops_info():
    img = Image.new("RGB", (20, 20))
    img.info["exif"] = b"fake"
    img.info["xmp"] = b"also_fake"
    clean = strip_metadata(img)
    assert "exif" not in clean.info or not clean.info.get("exif")

# --- film grain tests ---

def test_grain_changes_pixels():
    arr = np.full((50, 50, 3), 128, dtype=np.uint8)
    out = add_film_grain(arr, amount=10)
    assert not np.array_equal(arr, out)

def test_grain_preserves_range():
    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    out = add_film_grain(arr, amount=20)
    assert out.min() >= 0
    assert out.max() <= 255

def test_grain_zero_amount_no_change():
    arr = np.full((20, 20, 3), 100, dtype=np.uint8)
    out = add_film_grain(arr, amount=0)
    assert np.array_equal(arr, out)

# --- JPEG round-trip tests ---

def test_jpeg_roundtrip_returns_image():
    img = make_test_image((40, 40))
    out = jpeg_roundtrip(img, quality=85, passes=2)
    assert isinstance(out, Image.Image)
    assert out.size == img.size

def test_jpeg_roundtrip_no_crash_on_small():
    img = Image.new("RGB", (5, 5))
    out = jpeg_roundtrip(img, quality=60, passes=1)
    assert out.size == (5, 5)

# --- resize cycle tests ---

def test_resize_cycle_preserves_dimensions():
    img = make_test_image((80, 80))
    out = resize_cycle(img, factor=0.5, cycles=1)
    assert out.size == (80, 80)

def test_resize_cycle_alters_pixels():
    img = make_test_image((100, 100))
    out = resize_cycle(img, factor=0.5, cycles=1)
    assert not np.array_equal(np.array(img), np.array(out))

# --- DCT band perturbation tests ---

def test_dct_changes_pixels():
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    out = dct_band_perturb(arr, band_lo=0.4, band_hi=0.8, strength=0.5)
    assert not np.array_equal(arr, out)

def test_dct_preserves_shape():
    arr = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    out = dct_band_perturb(arr, band_lo=0.4, band_hi=0.8, strength=0.5)
    assert out.shape == arr.shape

def test_dct_zero_strength_no_change():
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    out = dct_band_perturb(arr, band_lo=0.3, band_hi=0.8, strength=0)
    assert np.array_equal(arr, out)

def test_dct_preserves_brightness():
    """DCT perturbation should not change overall brightness (DC untouched)."""
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = dct_band_perturb(arr, band_lo=0.4, band_hi=0.8, strength=0.5)
    # DC coefficient is untouched, brightness should stay ~same
    assert abs(float(out.mean()) - 128.0) < 8.0

def test_dct_preserves_low_freq():
    """Low frequency content (structure) should pass through mostly intact."""
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = dct_band_perturb(arr, band_lo=0.4, band_hi=0.8, strength=0.3)
    assert np.abs(out.astype(float) - arr.astype(float)).mean() < 10.0

# --- color jitter tests ---

def test_color_jitter_returns_image():
    img = make_test_image((40, 40))
    out = color_jitter(img, jitter=0.1, saturation=1.05)
    assert isinstance(out, Image.Image)
    assert out.size == img.size

# --- full pipeline tests ---

def test_process_image_returns_image():
    img = make_test_image((60, 60))
    out = process_image(img, strength="light")
    assert isinstance(out, Image.Image)

def test_process_image_all_strengths():
    for s in STRENGTH_PROFILES:
        img = make_test_image((50, 50))
        out = process_image(img, strength=s)
        assert out.size == (50, 50)

def test_process_image_invalid_strength():
    img = make_test_image((20, 20))
    try:
        process_image(img, strength="maximum")
        assert False, "should have raised"
    except ValueError:
        pass

def test_process_image_skip_all():
    img = make_test_image((30, 30))
    out = process_image(img, strength="light",
                        skip_metadata=True, skip_grain=True,
                        skip_jpeg=True, skip_resize=True,
                        skip_dct=True,
                        skip_color=True)
    # metadata strip still ran structure change, but pixels preserved
    assert out.size == (30, 30)

def test_process_image_different_from_input():
    img = make_test_image((80, 80))
    out = process_image(img, strength="heavy")
    assert not np.array_equal(np.array(img), np.array(out))

# --- file I/O tests ---

def test_process_file_writes_output():
    with tempfile.TemporaryDirectory() as tmp:
        inp = make_image_file(os.path.join(tmp, "test.jpg"))
        out = process_file(inp, strength="medium")
        assert os.path.exists(out)

def test_process_file_auto_name():
    with tempfile.TemporaryDirectory() as tmp:
        inp = make_image_file(os.path.join(tmp, "test.jpg"))
        out = process_file(inp)
        assert "_clean" in os.path.basename(out)

def test_process_file_invalid_path():
    try:
        process_file("nonexistent.jpg")
        assert False
    except (FileNotFoundError, OSError):
        pass

# --- batch tests ---

def test_batch_process():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            make_image_file(os.path.join(tmp, f"img{i}.jpg"))
        results = batch_process(tmp, strength="light")
        assert len(results) == 3
        for inp, out in results:
            assert "ERROR" not in out

def test_batch_create_output_dir():
    with tempfile.TemporaryDirectory() as tmp:
        make_image_file(os.path.join(tmp, "a.jpg"))
        outdir = os.path.join(tmp, "cleaned")
        results = batch_process(tmp, output_dir=outdir, strength="light")
        assert os.path.isdir(outdir)

# --- analyze tests ---

def test_analyze_metadata_returns_dict():
    with tempfile.TemporaryDirectory() as tmp:
        path = make_image_file(os.path.join(tmp, "test.jpg"))
        info = analyze(path)
        assert isinstance(info, dict)
        assert "has_exif" in info
        assert "format" in info

# --- CLI smoke test ---

def test_cli_smoke():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        inp = make_image_file(os.path.join(tmp, "test.jpg"))
        out = os.path.join(tmp, "out.jpg")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "undetect.py"),
             "process", inp, "-o", out, "--strength", "light"],
            capture_output=True, timeout=30
        )
        assert result.returncode == 0, result.stderr.decode()
        assert os.path.exists(out)

# --- edge cases ---

def test_tiny_image():
    img = Image.new("RGB", (3, 3))
    out = process_image(img, strength="heavy")
    assert out.size == (3, 3)

def test_grayscale_via_rgb():
    arr = np.full((20, 20, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    out = process_image(img, strength="light")
    assert out.mode == "RGB"

def test_uniform_image():
    img = Image.new("RGB", (50, 50), (255, 0, 0))
    out = process_image(img, strength="medium")
    assert out.size == (50, 50)
