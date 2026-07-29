# undetect

Alters AI-generated images to reduce AI detection scores. Drop in an image from Midjourney, DALL-E, Stable Diffusion, or any diffusion model output, and the tool applies six processing techniques that disrupt the statistical signatures detectors look for.

Live demo: https://saman-ghorayshi.github.io/undetect/

## How it works

AI image detectors (Hive, Sightengine, Illuminarty, SynthID-reading tools) identify machine-generated images through several vectors:

- **Metadata**: EXIF tags, C2PA provenance watermarks, XMP packets
- **Pixel patterns**: over-smoothness, unnaturally regular noise distributions
- **Frequency domain**: DCT coefficient signatures from the diffusion process
- **Spatial features**: consistent texture regularity, lack of organic imperfections

This tool applies a six-stage pipeline that disrupts each vector:

1. **Metadata strip** - rebuilds pixel data into a fresh image object, dropping all EXIF/XMP/C2PA/IPTC tags
2. **Color jitter** - random brightness, contrast, and saturation shifts break the unnaturally consistent color histograms AI produces
3. **Resize cycle** - downscale then upscale with Lanczos resampling destroys fine pixel-level patterns
4. **Film grain** - adds organic monochromatic noise (Gaussian, shaped to mimic analog film) that masks the smooth micro-pixel patterns from diffusion
5. **Adversarial noise** - uniform pixel perturbations in a controlled range that shift classifier confidence scores without visible degradation
6. **JPEG round-trip** - re-encodes through JPEG multiple times, scrambling DCT coefficients in the frequency domain

Three strength presets (light/medium/heavy) trade image quality for detection disruption. Heavy is for when you need maximum bypass and don't care about artifacting.

## Install

```
pip install Pillow numpy piexif
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
python undetect.py process image.png --skip-fft --skip-resize --strength heavy
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

| Image | Before | After (heavy) | Classification |
|-------|--------|---------------|-----------------|
| Synthetic gradient | 56.5% AI | 38.8% Real | flipped |
| Real AI portrait | 97.0% AI | 22.9% Real | flipped |

The detector's own indicators flipped from "overly perfect symmetry, unnatural lighting, blurred edges" to "natural skin texture, organic shadow falloff, authentic depth of field."

## Known limitations

- The FFT frequency domain method can introduce visible ringing artifacts on images with strong high-contrast edges. It only runs on heavy strength.
- JPEG round-trip is lossy by design. Each pass reduces quality. The tool caps at 3 passes (heavy) to avoid excessive degradation.
- Resize cycle with heavy strength (factor 0.55) can soften small details. Use light (0.85) if preserving texture matters.
- The browser demo skips the FFT stage because it's too slow on large images in JavaScript. The Python CLI runs all stages.
- This tool disrupts common detection vectors, but adversarial methods are an arms race. New detector models may adapt to these exact techniques.
- Metadata stripping is intentional. If you need provenance tracking (e.g. C2PA for legal compliance), this tool is not for you.

## License

MIT
