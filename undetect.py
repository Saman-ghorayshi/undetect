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

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False


STRENGTH_PROFILES = {
    "light": {
        "grain_amount": 5,
        "jpeg_quality": 88,
        "jpeg_roundtrips": 2,
        "resize_factor": 0.82,
        "resize_cycles": 1,
        "noise_strength": 3,
        "fft_cutoff": 0.4,
        "color_jitter": 0.05,
        "saturation_shift": 1.02,
    },
    "medium": {
        "grain_amount": 6,
        "jpeg_quality": 85,
        "jpeg_roundtrips": 2,
        "resize_factor": 0.75,
        "resize_cycles": 1,
        "noise_strength": 3,
        "fft_cutoff": 0.3,
        "color_jitter": 0.06,
        "saturation_shift": 1.03,
    },
    "heavy": {
        "grain_amount": 8,
        "jpeg_quality": 82,
        "jpeg_roundtrips": 2,
        "resize_factor": 0.78,
        "resize_cycles": 1,
        "noise_strength": 8,
        "fft_cutoff": 0.3,
        "color_jitter": 0.07,
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


def add_film_grain(arr, amount=8):
    """Gaussian noise, monochromatic. Masks diffusion smoothness."""
    noise = np.random.normal(0, amount, arr.shape[:2])
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


def adversarial_noise(arr, strength=4):
    """Uniform per-pixel perturbation. Shifts classifier scores."""
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    out = np.clip(arr.astype(np.int16) + noise, 0, 255)
    return out.astype(np.uint8)


def fft_lowpass(arr, cutoff=0.3):
    """
    Butterworth lowpass in frequency domain. Attenuates high-freq AI
    artifacts while keeping DC (brightness).  cutoff is fraction of
    Nyquist radius; smaller = more aggressive.

    Per the arxiv 2410.01574 paper, frequency-domain perturbation
    drops detector AUC by 40-60%. A smooth lowpass avoids the ringing
    artifacts a hard cutoff introduces.
    """
    if cutoff <= 0:
        return arr
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    # normalized distance from center (DC term)
    yy = np.arange(h) - cy
    xx = np.arange(w) - cx
    yy, xx = np.meshgrid(yy, xx, indexing='ij')
    radius = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
    # butterworth: 1 / (1 + (r/r0)^4). r0 is the -3dB point
    r0 = cutoff
    mask = 1.0 / (1.0 + (radius / r0) ** 4)

    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        channel = arr[:, :, c].astype(np.float32)
        fft = np.fft.fftshift(np.fft.fft2(channel))
        fft = fft * mask
        channel_back = np.fft.ifft2(np.fft.ifftshift(fft)).real
        out[:, :, c] = channel_back
    return np.clip(out, 0, 255).astype(np.uint8)


def color_jitter(img, jitter=0.06, saturation=1.03):
    """Contrast/saturation only, no brightness nudge. Avoids darkening."""
    contrast = 1.0 + np.random.uniform(-jitter, jitter)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img


def process_image(img, strength="medium", skip_metadata=False,
                  skip_grain=False, skip_jpeg=False,
                  skip_resize=False, skip_noise=False,
                  skip_fft=False, skip_color=False):
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
    if not skip_noise:
        arr = adversarial_noise(arr, cfg["noise_strength"])
    if not skip_fft and cfg["fft_cutoff"] > 0:
        arr = fft_lowpass(arr, cfg["fft_cutoff"])

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
    p_proc.add_argument("--skip-noise", action="store_true")
    p_proc.add_argument("--skip-fft", action="store_true")
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
            skip_noise=args.skip_noise,
            skip_fft=args.skip_fft,
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
