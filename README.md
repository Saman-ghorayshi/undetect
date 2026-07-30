# undetect

Alters AI-generated images to reduce AI detection scores. Drop in an image from Midjourney, DALL-E, Stable Diffusion, or any diffusion model output, and the tool applies processing techniques that disrupt the statistical signatures detectors look for.

Live demo: https://saman-ghorayshi.github.io/undetect/

## How it works

AI image detectors (Hive, Sightengine, DeepAI, Illuminarty, SynthID readers) flag machine-generated images through several vectors:

- **Metadata**: EXIF tags, C2PA provenance watermarks, XMP packets
- **Pixel statistics**: over-smooth gradients, unnaturally uniform noise distribution
- **Frequency domain**: DCT coefficient signatures left by the diffusion process
- **Texture regularity**: consistent micro-patterns, lack of organic imperfections

Research shows generated images have sparse high-frequency content compared to real images, which have rich high-frequency detail (FBA2D, arxiv 2512.09264). Det exploit this spectral difference. This tool targets the specific frequency bands where AI fingerprints concentrate.

The pipeline:

1. **Metadata strip** - rebuilds pixel data into a fresh image object, dropping EXIF/XMP/C2PA/IPTC
2. **Color jitter** - small random contrast and saturation shifts (brightness only goes up, so the image never darkens)
3. **Resize cycle** - downscale then upscale with Lanczos resampling, breaks pixel-level patterns
4. **Film grain** - Gaussian monochromatic noise weighted by inverse luminance AND inverse gradient. More grain in dark flat areas (where detectors look for AI smoothness), less in bright edge/detail areas (so fabric texture and skin pores survive)
5. **DCT band perturbation** - decomposes each 8x8 block into DCT coefficients and selectively attenuates + perturbs the mid-high frequency bands where AI fingerprints live. Leaves DC and low frequencies (brightness, structure) and very high frequencies (fine detail) untouched. Based on the frequency-band attack approach from FBA2D (arxiv 2512.09264) and the low-frequency preservation principle from ERASE (IEEE TDSC 2026)
6. **JPEG round-trip** - re-encodes through JPEG 1-2 times, scrambling DCT coefficients

Three strength presets (light/medium/heavy) trade image quality for detection disruption. Each preset was tuned and tested against DeepAI's detector.

## Install

```
pip install Pillow numpy scipy piexif
```

## Usage

Process a single image:
```
python undetect.py process image.png -o clean.jpg --strength heavy
```

Simple mode (defaults to medium):
```
python undetect.py image.png
```

Batch process a folder:
```
python undetect.py batch ./ai_images/ -o ./clean/ --strength medium
```

Check what metadata an image carries:
```
python undetect.py analyze image.png
```

Skip specific stages (for testing individual techniques):
```
python undetect.py process image.png --skip-dct --skip-resize --strength heavy
```

## Python API

```python
from undetect import process_image, load_image

img = load_image("ai_generated.png")
clean = process_image(img, strength="heavy")
clean.save("clean.jpg", quality=88)
```

## Verified Results

Tested against DeepAI's AI Image Detector (https://deepai.org/ai-image-detector):

| Image | Before | After (medium) | Classification |
|-------|--------|----------------|-----------------|
| Synthetic gradient | 56.5% AI | 38.8% Real | flipped |
| Real AI portrait | 97.0% AI | 19.0% Real | flipped |

The detector's own indicators flipped from "overly perfect symmetry, unnatural lighting, blurred edges" to "natural skin texture, organic shadow falloff, authentic depth of field."

Quality metrics for the real AI portrait (506x675 JPEG):
- Medium: PSNR 32.0dB, brightness delta -1.4 (original mean 62.7, output 61.3)
- Heavy: PSNR 31.0dB, brightness delta -1.2

## Known limitations

- The DCT band perturbation introduces subtle coefficient changes that are invisible at normal viewing but may be detectable by statistical analysis tools.
- JPEG round-trip is lossy by design. Each pass reduces quality. The tool caps at 2 passes to avoid excessive degradation.
- Resize cycle with heavy strength (factor 0.75) can soften small details. Use light (0.85) if preserving texture matters.
- The browser demo skips the DCT stage because it requires scipy. The Python CLI runs all stages.
- This tool disrupts common detection vectors, but adversarial methods are an arms race. New detector models may adapt to these exact techniques.
- Metadata stripping is intentional. If you need provenance tracking (e.g. C2PA for legal compliance), this tool is not for you.

## License

MIT
