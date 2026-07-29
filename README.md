# undetect

Alters AI-generated images to reduce AI detection scores. Drop in an image from Midjourney, DALL-E, Stable Diffusion, or any diffusion model output, and the tool applies six processing techniques that disrupt the statistical signatures detectors look for.

Live demo: https://saman-ghorayshi.github.io/undetect/demo/

## How it works

AI image detectors (Hive, Sightengine, DeepAI, Illuminarty, SynthID readers) flag machine-generated images through several vectors:

- **Metadata**: EXIF tags, C2PA provenance watermarks, XMP packets
- **Pixel statistics**: over-smooth gradients, unnaturally uniform noise distribution
- **Frequency domain**: DCT coefficient signatures left by the diffusion process
- **Texture regularity**: consistent micro-patterns, lack of organic imperfections

This tool applies a 7-stage pipeline that disrupts each vector:

1. **Metadata strip** - rebuilds pixel data into a fresh image object, dropping EXIF/XMP/C2PA/IPTC
2. **Color jitter** - small random contrast and saturation shifts (no brightness change, so the image never darkens)
3. **Resize cycle** - downscale then upscale with Lanczos resampling, breaks pixel-level patterns
4. **Film grain** - Gaussian monochromatic noise shaped like analog film, masks diffusion smoothness
5. **Adversarial noise** - uniform per-pixel perturbation in a controlled range, shifts classifier confidence
6. **FFT lowpass** - Butterworth low-pass filter in the frequency domain, attenuates high-frequency artifacts without ringing (uses butterworth, not hard cut, so DC brightness is preserved)
7. **JPEG round-trip** - re-encodes through JPEG 1-3 times, scrambling DCT coefficients

Three strength presets (light/medium/heavy) trade image quality for detection disruption. Each preset was tuned and tested against DeepAI's detector.

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
