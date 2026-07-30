"""
undetect - make AI-generated images harder for detectors to flag.

Pipeline order and what each stage targets:
1. strip metadata    -> EXIF/XMP/C2PA/IPTC provenance
2. color jitter      -> histogram consistency (contrast/sat only, no brightness)
3. resize cycle      -> pixel-level fingerprint, texture regularity
4. film grain        -> smooth micro-pixel patterns from diffusion
5. adversarial noise -> classifier confidence score shift
6. fft lowpass       -> frequency-domain artifact signatures
7. jpeg roundtrip    -> DCT coefficient scrambling (most effective single stage)
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from scipy.fft import dctn, idctn

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False


STRENGTH_PROFILES = {
    "light": {
        "grain_amount": 4,
        "jpeg_quality": 88,
        "jpeg_roundtrips": 2,
        "resize_factor": 0.85,
        "resize_cycles": 1,
        "dct_band_lo": 0.40,
        "dct_band_hi": 0.85,
        "dct_strength": 0.5,
        "color_jitter": 0.04,
        "saturation_shift": 1.02,
    },
    "medium": {
        "grain_amount": 6,
        "jpeg_quality": 85,
        "jpeg_roundtrips": 2,
        "resize_factor": 0.80,
        "resize_cycles": 1,
        "dct_band_lo": 0.30,
        "dct_band_hi": 0.85,
        "dct_strength": 0.7,
        "color_jitter": 0.05,
        "saturation_shift": 1.03,
    },
    "heavy": {
        "grain_amount": 8,
        "jpeg_quality": 82,
        "jpeg_roundtrips": 2,
        "resize_factor": 0.75,
        "resize_cycles": 1,
        "dct_band_lo": 0.30,
        "dct_band_hi": 0.85,
        "dct_strength": 0.75,
        "color_jitter": 0.06,
        "saturation_shift": 1.04,
    },
}


def load_image(path):
    if isinstance(path, (str, Path)):
        img = Image.open(path)
    elif isinstance(path, Image.Image):
        img = path
    else:
        raise TypeError(f"expected path or PIL Image, got {type(path)}")
    return img.convert("RGB")


def strip_metadata(img):
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata()))
    return clean


def add_film_grain(arr, amount=8, luma_weight=True):
    """
    Gaussian noise, monochromatic. Masks diffusion smoothness.

    When luma_weight=True, noise amplitude scales with the inverse luminance
    of each pixel — more grain in dark areas (where detectors look for
    smooth gradient evidence) and less in bright/detail areas (so fabric
    texture and skin detail survive). Mirrors analog film which has
    more grain in shadows than highlights.
    """
    noise = np.random.normal(0, amount, arr.shape[:2])
    if luma_weight:
        # weights: 1.0 in deep shadows, gradually less up to 0.2 in highlights
        luma = np.array(arr).astype(np.float32).mean(axis=2)
        # also scale by inverse gradient: more noise in flat areas, less in edges
        gx = np.gradient(luma, axis=1)
        gy = np.gradient(luma, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_norm = np.clip(grad_mag / (grad_mag.max() + 1e-6), 0, 1)
        luma_weight_map = 1.0 - 0.8 * np.clip(luma / 255.0, 0, 1)
        edge_weight = 1.0 - 0.7 * grad_norm  # less noise on edges
        noise = noise * luma_weight_map * edge_weight
    noise = noise[:, :, np.newaxis]
    if arr.ndim == 3:
        noise = np.repeat(noise, arr.shape[2], axis=2)
    out = np.clip(arr.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def jpeg_roundtrip(img, quality=85, passes=2):
    """Re-encode JPEG N times. Scrambles DCT coefficients detectors read."""
    current = img
    for _ in range(passes):
        buf = io.BytesIO()
        current.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        current = Image.open(buf)
        current.load()
    return current


def resize_cycle(img, factor=0.72, cycles=1):
    """Down then up with Lanczos. Breaks pixel-level spatial fingerprints."""
    w, h = img.size
    current = img
    for _ in range(cycles):
        sw = max(1, int(w * factor))
        sh = max(1, int(h * factor))
        current = current.resize((sw, sh), Image.LANCZOS)
        current = current.resize((w, h), Image.LANCZOS)
    return current


def dct_band_perturb(arr, band_lo=0.4, band_hi=0.8, strength=0.5):
    """
    DCT-domain frequency-band perturbation. Targets the specific frequency
    bands where AI-generated fingerprints concentrate (mid-high per FBA2D,
    arxiv 2512.09264) without touching the DC/low bands (brightness,
    structure) or the very high bands (fine detail).

    Two operations in the target band:
    - attenuate: scale coefficients down by `strength` (removes the AI
      fingerprint energy the way a lowpass does, but surgical)
    - perturb: add small random noise to break regularity

    Works on 8x8 blocks (same as JPEG DCT structure).
    """
    if strength <= 0:
        return arr
    out = np.zeros_like(arr, dtype=np.float32)
    h, w = arr.shape[:2]

    ucord = np.arange(8)
    uy, ux = np.meshgrid(ucord, ucord, indexing='ij')
    freq = np.sqrt((ux / 8.0)**2 + (uy / 8.0)**2)
    band_mask = (freq >= band_lo) & (freq <= band_hi)

    for c in range(arr.shape[2]) if arr.ndim == 3 else [0]:
        channel = arr[:, :, c] if arr.ndim == 3 else arr
        ch = channel.astype(np.float32).copy()
        for y in range(0, h - 7, 8):
            for x in range(0, w - 7, 8):
                block = ch[y:y+8, x:x+8]
                coeffs = dctn(block, type=2, norm='ortho')
                # attenuate target band (reduce fingerprint energy)
                atten = np.ones((8, 8), dtype=np.float32)
                atten[band_mask] = 1.0 - strength
                # add random perturbation to break regularity
                noise = np.zeros((8, 8), dtype=np.float32)
                noise[band_mask] = np.random.normal(0, strength * 2,
                                                     size=band_mask.sum())
                coeffs = coeffs * atten + noise
                ch[y:y+8, x:x+8] = idctn(coeffs, type=2, norm='ortho')
        if arr.ndim == 3:
            out[:, :, c] = ch
        else:
            out = ch

    return np.clip(out, 0, 255).astype(np.uint8)


def color_jitter(img, jitter=0.06, saturation=1.03):
    """Contrast/saturation only, no brightness nudge. Avoids darkening."""
    contrast = 1.0 + np.random.uniform(-jitter, jitter)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img


def process_image(img, strength="medium", skip_metadata=False,
                  skip_grain=False, skip_jpeg=False,
                  skip_resize=False, skip_dct=False,
                  skip_color=False):
    if strength not in STRENGTH_PROFILES:
        raise ValueError(f"strength must be one of {list(STRENGTH_PROFILES)}")
    cfg = STRENGTH_PROFILES[strength]

    if not skip_metadata:
        img = strip_metadata(img)
    if not skip_color:
        img = color_jitter(img, cfg["color_jitter"], cfg["saturation_shift"])
    if not skip_resize:
        img = resize_cycle(img, cfg["resize_factor"], cfg["resize_cycles"])

    arr = np.array(img)

    if not skip_grain:
        arr = add_film_grain(arr, cfg["grain_amount"])
    if not skip_dct and cfg["dct_strength"] > 0:
        arr = dct_band_perturb(arr, cfg["dct_band_lo"], cfg["dct_band_hi"],
                               cfg["dct_strength"])

    img = Image.fromarray(arr)

    if not skip_jpeg:
        img = jpeg_roundtrip(img, cfg["jpeg_quality"], cfg["jpeg_roundtrips"])

    return img


def process_file(input_path, output_path=None, strength="medium", **kwargs):
    img = load_image(input_path)
    result = process_image(img, strength=strength, **kwargs)
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_clean{p.suffix or '.jpg'}"
    else:
        output_path = Path(output_path)
    if str(output_path).lower().endswith(('.png', '.webp', '.tiff', '.bmp')):
        result.save(output_path)
    else:
        result.save(output_path, format="JPEG", quality=90)
    return str(output_path)


def batch_process(input_dir, output_dir=None, strength="medium", **kwargs):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")
    if output_dir is None:
        output_dir = input_dir / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    results = []
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in exts and f.is_file():
            out = output_dir / f"{f.stem}_clean.jpg"
            try:
                process_file(f, out, strength=strength, **kwargs)
                results.append((str(f), str(out)))
            except Exception as e:
                results.append((str(f), f"ERROR: {e}"))
    return results


def analyze(img_or_path):
    if isinstance(img_or_path, (str, Path)):
        img = Image.open(img_or_path)
    else:
        img = img_or_path
    findings = {
        "has_exif": bool(img.info.get("exif")),
        "has_xmp": "xmp" in img.info or "XML:com.adobe.xmp" in img.info,
        "has_icc": "icc_profile" in img.info,
        "has_transparency": "transparency" in img.info,
        "info_keys": list(img.info.keys()),
        "format": img.format,
        "size": img.size,
        "mode": img.mode,
    }
    if HAS_PIEXIF and findings["has_exif"]:
        try:
            exif_dict = piexif.load(img.info["exif"])
            findings["exif_tags"] = {
                k: str(v)[:100] for k, v in exif_dict.items()
                if isinstance(v, dict) and v
            }
        except Exception:
            findings["exif_tags"] = "unreadable"
    return findings


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="undetect",
        description="alter AI-generated images to reduce detection scores",
    )
    sub = parser.add_subparsers(dest="command")

    p_proc = sub.add_parser("process", help="process a single image")
    p_proc.add_argument("input", help="input image path")
    p_proc.add_argument("-o", "--output")
    p_proc.add_argument("-s", "--strength", choices=list(STRENGTH_PROFILES), default="medium")
    p_proc.add_argument("--skip-metadata", action="store_true")
    p_proc.add_argument("--skip-grain", action="store_true")
    p_proc.add_argument("--skip-jpeg", action="store_true")
    p_proc.add_argument("--skip-resize", action="store_true")
    p_proc.add_argument("--skip-dct", action="store_true")
    p_proc.add_argument("--skip-color", action="store_true")

    p_batch = sub.add_parser("batch", help="process all images in a directory")
    p_batch.add_argument("input_dir")
    p_batch.add_argument("-o", "--output-dir")
    p_batch.add_argument("-s", "--strength", choices=list(STRENGTH_PROFILES), default="medium")

    p_info = sub.add_parser("analyze", help="show metadata info for an image")
    p_info.add_argument("input")

    args, extra = parser.parse_known_args()
    if args.command is None and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        out = process_file(sys.argv[1], strength="medium")
        print(f"saved: {out}")
        return

    if args.command == "process":
        out = process_file(
            args.input, args.output, args.strength,
            skip_metadata=args.skip_metadata,
            skip_grain=args.skip_grain,
            skip_jpeg=args.skip_jpeg,
            skip_resize=args.skip_resize,
            skip_dct=args.skip_dct,
            skip_color=args.skip_color,
        )
        print(f"saved: {out}")
    elif args.command == "batch":
        results = batch_process(args.input_dir, args.output_dir, args.strength)
        for inp, out in results:
            print(f"  {inp} -> {out}")
        ok = sum(1 for _, o in results if "ERROR" not in o)
        print(f"done: {ok}/{len(results)} ok")
    elif args.command == "analyze":
        import json
        info = analyze(args.input)
        print(json.dumps(info, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
