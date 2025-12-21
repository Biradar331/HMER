"""
Generate a .pkl mapping uid -> image ndarray (channels, H, W).
Usage (from project root):
    python gen_pkl.py --image-dir off_image_test --caption-file test_caption.txt --out-file offline-test.pkl --channels 1 --ext bmp
"""
from pathlib import Path
import argparse
import pickle
import numpy as np
from PIL import Image

# Try to import image reading backend (imageio v2 preferred, fallback to imageio)
try:
    import imageio.v2 as imageio
except Exception:
    try:
        import imageio  # older imageio
    except Exception:
        imageio = None  # will fall back to PIL

def imread_fallback(path: Path) -> np.ndarray:
    """Read image as 2D grayscale numpy array using imageio if available, else PIL."""
    if imageio is not None:
        try:
            im = imageio.imread(str(path))
        except Exception:
            im = np.asarray(Image.open(str(path)).convert("L"))
    else:
        im = np.asarray(Image.open(str(path)).convert("L"))
    # If color, convert to grayscale
    if im.ndim == 3:
        im = np.asarray(Image.fromarray(im).convert("L"))
    return im

def build_features(image_dir: Path, caption_file: Path, channels: int, ext: str):
    features = {}
    lines = caption_file.read_text(encoding="utf-8").splitlines()
    total = 0
    processed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        uid = parts[0]
        total += 1

        imgs = []
        missing = False
        h = w = None
        for ch in range(channels):
            # try uid_ch.ext first, then uid.ext for single-channel datasets
            img_path = image_dir / f"{uid}_{ch}.{ext}"
            if not img_path.is_file() and channels == 1:
                alt = image_dir / f"{uid}.{ext}"
                if alt.is_file():
                    img_path = alt
            if not img_path.is_file():
                missing = True
                break
            try:
                im = imread_fallback(img_path)
            except Exception as e:
                print("Warning: failed to read", img_path, ":", e)
                missing = True
                break
            if h is None:
                h, w = im.shape
            else:
                if im.shape != (h, w):
                    im = np.asarray(Image.fromarray(im).resize((w, h)))
            imgs.append(im.astype(np.uint8))

        if missing:
            print(f"Warning: skipping uid={uid} (missing/unreadable images)")
            continue

        arr = np.stack(imgs, axis=0)  # (channels, H, W)
        features[uid] = arr
        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} / {total} examples...")

    print(f"Finished. Listed: {total}, processed: {processed}, skipped: {total-processed}")
    return features

def main():
    parser = argparse.ArgumentParser(description="Generate .pkl of images (id -> ndarray)")
    parser.add_argument("--image-dir", type=Path, default=Path("off_image_test"))
    parser.add_argument("--caption-file", type=Path, default=Path("test_caption.txt"))
    parser.add_argument("--out-file", type=Path, default=Path("offline-test.pkl"))
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--ext", type=str, default="bmp")
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    caption_file = args.caption_file.resolve()
    out_file = args.out_file.resolve()

    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")
    if not caption_file.is_file():
        raise SystemExit(f"Caption file not found: {caption_file}")

    features = build_features(image_dir, caption_file, args.channels, args.ext)

    with out_file.open("wb") as fp:
        pickle.dump(features, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(features)} entries to {out_file}")

if __name__ == "__main__":
    main()