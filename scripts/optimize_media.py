#!/usr/bin/env python3
"""Regenerate committed web media with cwebp, ffmpeg, and ffprobe installed.

Run from any directory: python3 scripts/optimize_media.py
Original photographs and GIFs remain available at their existing URLs.
"""

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT)


def dimensions(path):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path),
    ]))
    stream = info["streams"][0]
    return stream["width"], stream["height"]


def webp(source, target, width, quality=82, crop=None):
    args = ["cwebp", "-quiet", "-q", quality, "-m", 6, "-metadata", "none"]
    if crop:
        args += ["-crop", *crop]
    run(*args, "-resize", width, 0, source, "-o", target)


def main():
    images = ROOT / "images/optimized"
    videos = ROOT / "files/optimized"
    images.mkdir(exist_ok=True)
    videos.mkdir(exist_ok=True)
    manifest = {"images": {}, "animations": {}}
    stills = [
        # Removing 1,350 vertical pixels at y=864 preserves the existing 64%
        # object-position crop while avoiding downloads of unused sky/trail.
        ("tetons", "images/tetons_wall.jpg", [480, 960, 1672], 80, [0, 864, 1672, 1150]),
        ("headshot", "images/headshot.jpeg", [165, 330, 660], 85, None),
        ("uncertainty-course", "images/uncertainty-course-screenshot.png", [576, 1152], 88, None),
        ("classroom", "images/stat6130_lecture.jpg", [340, 680], 82, None),
        ("dynamix", "images/DynamIX_album_art.png", [200, 400], 85, None),
    ]
    for name, source, widths, quality, crop in stills:
        variants = []
        for width in widths:
            target = images / f"{name}-{width}.webp"
            webp(source, target, width, quality, crop)
            w, h = dimensions(target)
            variants.append({"url": "/" + str(target.relative_to(ROOT)), "width": w, "height": h})
        manifest["images"][name] = {"original": "/" + source, "variants": variants}
        print(f"{name}: {Path(ROOT / source).stat().st_size:,} bytes → "
              + ", ".join(f"{v['width']}w {Path(ROOT / v['url'].lstrip('/')).stat().st_size:,} bytes" for v in variants))

    animations = [
        ("regularization", "files/regularization-path.gif", 1200, 25, 18),
        ("spectrum", "files/exp_animation.gif", 900, 12, 18),
        ("kick-return", "files/bdb.gif", 1440, 30, 22),
    ]
    with tempfile.TemporaryDirectory() as temp:
        for name, source, output_width, fps, crf in animations:
            width, height = dimensions(ROOT / source)
            duration = float(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(ROOT / source),
            ]))
            target = videos / f"{name}.mp4"
            # GIF transparency must be composited onto white before encoding;
            # simply dropping alpha can make the spectrum's black axes vanish.
            filters = (f"color=c=white:s={width}x{height}:r={fps}[bg];"
                       f"[bg][0:v]overlay=shortest=1:format=auto,"
                       f"scale={output_width}:-2:flags=lanczos,setsar=1,"
                       f"fps={fps},format=yuv420p,"
                       f"tpad=stop_mode=clone:stop_duration={duration}[v]")
            run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ignore_loop", 1, "-i", source, "-filter_complex", filters,
                "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "slow",
                "-crf", crf, "-t", duration, "-movflags", "+faststart", target)
            encoded_duration = float(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(target),
            ]))
            assert abs(encoded_duration - duration) <= 1 / fps + .001, name
            png = Path(temp) / f"{name}.png"
            run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", target,
                "-frames:v", 1, png)
            poster = images / f"{name}-poster.webp"
            webp(png, poster, min(output_width, 960) if name == "kick-return" else output_width,
                 80 if name == "kick-return" else 88)
            w, h = dimensions(target)
            manifest["animations"][name] = {
                "url": "/" + str(target.relative_to(ROOT)),
                "poster": "/" + str(poster.relative_to(ROOT)),
                "original": "/" + source, "width": w, "height": h,
            }
            print(f"{name}: {(ROOT / source).stat().st_size:,} bytes → "
                  f"{target.stat().st_size:,} bytes + {poster.stat().st_size:,}-byte poster")

    (ROOT / "_data/media.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
