"""Build the two-minute demo video.

Nothing here is a mock-up. The terminal scenes replay `/tmp/demo_out.txt`, which is the
verbatim stdout of `python demo.py` against a live Gemma 4 E4B - the same run whose output
appears in the notebook. What this script adds is pacing and narration, not content.

Frames are emitted as a handful of key images with explicit durations rather than at a
fixed frame rate: a terminal reveal only changes when a line appears, so rendering 30
identical PNGs a second would be a waste of both time and disk.

    MINIMAX_KEY=$(cat ~/.config/kaggle-agent/minimax_key) python3 scripts_video.py

Requires: ffmpeg, PIL. Narration via MiniMax speech-02-hd, falling back to macOS `say`
so the video can still be built without a key.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
BUILD = ROOT / "build_video"
ASSETS = ROOT / "assets"
DEMO = pathlib.Path("/tmp/demo_out.txt")

W, H = 1920, 1080
FPS = 30

INK = (250, 247, 243)
DIM = (150, 140, 130)
ACCENT = (214, 90, 40)
BLUE = (110, 150, 200)
GREEN = (130, 175, 130)
BG = (18, 16, 14)

MONO = "/System/Library/Fonts/Menlo.ttc"
DEVA = "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"
SANS = "/System/Library/Fonts/Supplemental/Helvetica.ttc"

VOICE = "English_Graceful_Lady"
MODEL = "speech-02-hd"

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

_font_cache: dict = {}


def font(path: str, size: int):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def is_deva(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ"


def runs(text: str):
    """Split a line into same-script runs so each can use a font that has the glyphs."""
    if not text:
        return []
    out, cur, cur_d = [], "", is_deva(text[0])
    for ch in text:
        d = is_deva(ch)
        if d != cur_d and ch != " ":
            out.append((cur, cur_d)); cur, cur_d = ch, d
        else:
            cur += ch
    out.append((cur, cur_d))
    return out


def draw_line(d: ImageDraw.ImageDraw, xy, text: str, size: int, fill) -> None:
    x, y = xy
    for run, deva in runs(text):
        f = font(DEVA if deva else MONO, int(size * (1.12 if deva else 1)))
        d.text((x, y - (size * 0.14 if deva else 0)), run, font=f, fill=fill)
        x += d.textlength(run, font=f)


def colour_for(line: str):
    s = line.strip()
    if s.startswith("ASK"):
        return ACCENT
    if s.startswith("SHE"):
        return INK
    if s.startswith("(asked because") or s.startswith("extracted:"):
        return DIM
    if s.startswith("[PASS]") or s.startswith("- "):
        return GREEN
    if s.startswith("==="):
        return BLUE
    if s.startswith("--"):
        return DIM
    return INK


def terminal_frame(lines: list[str], title: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 74], fill=(30, 27, 24))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([40 + i * 34, 28, 58 + i * 34, 46], fill=c)
    d.text((160, 26), title, font=font(MONO, 24), fill=DIM)

    y, size = 116, 27
    for line in lines[-30:]:
        draw_line(d, (60, y), line[:118], size, colour_for(line))
        y += int(size * 1.52)
    return img


def title_frame(big: list[str], small: str = "", deva: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 300
    if deva:
        d.text((150, 200), deva, font=font(DEVA, 62), fill=INK)
        y = 360
    for i, line in enumerate(big):
        d.text((150, y), line, font=font(SANS, 76),
               fill=ACCENT if i and i == len(big) - 1 else INK)
        y += 104
    if small:
        for j, sl in enumerate(small.split("\n")):
            d.text((150, y + 46 + j * 50), sl, font=font(SANS, 34), fill=DIM)
    return img


def image_frame(path: pathlib.Path) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    src = Image.open(path).convert("RGB")
    scale = min((W - 200) / src.width, (H - 200) / src.height)
    src = src.resize((int(src.width * scale), int(src.height * scale)), Image.LANCZOS)
    img.paste(src, ((W - src.width) // 2, (H - src.height) // 2))
    return img


# ----------------------------------------------------------------- narration

def tts(text: str, out: pathlib.Path) -> None:
    key = os.environ.get("MINIMAX_KEY", "").strip()
    if key:
        body = json.dumps({
            "model": MODEL, "text": text, "stream": False,
            "voice_setting": {"voice_id": VOICE, "speed": 1.12, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"},
        }).encode()
        req = urllib.request.Request(
            "https://api.minimax.chat/v1/t2a_v2", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            audio = d.get("data", {}).get("audio")
            if audio:
                out.write_bytes(bytes.fromhex(audio))
                return
            print(f"  ! minimax: {d.get('base_resp')}", file=sys.stderr)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! minimax failed ({exc}), falling back to `say`", file=sys.stderr)
    aiff = out.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), str(out)],
                   check=True)
    aiff.unlink(missing_ok=True)


def duration(path: pathlib.Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# ----------------------------------------------------------------- scenes

def demo_lines() -> list[str]:
    return DEMO.read_text().splitlines()


def build_scenes() -> list[dict]:
    L = demo_lines()

    def seg(a, b):
        return [x for x in L[a:b]]

    ask_start = next(i for i, l in enumerate(L) if l.startswith("ASK"))
    entitled = next(i for i, l in enumerate(L) if "ENTITLED TO APPLY" in l)
    docs = next(i for i, l in enumerate(L) if "DOCUMENTS TO COLLECT" in l)
    audit = next(i for i, l in enumerate(L) if "AUDIT TRAIL" in l)
    msg = next(i for i, l in enumerate(L) if "MESSAGE TO THE PERSON" in l)

    # Narration is budgeted, not written and then trimmed: roughly 145 words a minute,
    # and the whole thing has to land under two minutes.
    return [
        dict(kind="title", deva="योजना सारथी",
             big=["The scheme is not the hard part.", "The interview is."],
             small="An offline welfare-entitlement navigator\nbuilt on Gemma 4 E4B",
             say="Uttar Pradesh publishes its welfare rules. They are not secret. Anyone "
                 "with the circulars and an hour could work out what a family is owed. "
                 "Nobody has the hour."),
        dict(kind="image", path=ASSETS / "pipeline.png",
             say="Gemma works at the edges. It turns what she says into twenty-one "
                 "declared fields, then writes the answer back in her language. In "
                 "between, a rule engine an officer can audit. Gemma never decides who "
                 "qualifies."),
        dict(kind="term", lines=seg(0, ask_start), title="python demo.py",
             say="A real run. She speaks once, in Hindi. My husband died last year, two "
                 "daughters in school, kutcha house. Gemma pulls nine fields out of that "
                 "one sentence."),
        dict(kind="term", lines=seg(ask_start, entitled - 2), title="python demo.py",
             reveal=3,
             say="Now the engine chooses. It scores every unknown field by the money "
                 "knowing it unlocks, and opens with the ration card — that one fact "
                 "gates thirty thousand rupees and a five lakh health cover. Seven "
                 "questions, then it stops."),
        dict(kind="image", path=ASSETS / "interview_curve.png",
             say="Measured, not asserted. Across eight households, after one question, "
                 "fifty-two percent of what the average family is owed is established. A "
                 "paper form has nothing until question four. That matters, because "
                 "interviews get abandoned."),
        dict(kind="term", lines=seg(entitled, audit), title="python demo.py", reveal=6,
             say="Six schemes — about six and three quarter lakh, plus a thousand a "
                 "month. Documents collapsed across them, so she fetches her Aadhaar "
                 "once, not six times."),
        dict(kind="term", lines=seg(audit, len(L)), title="python demo.py", reveal=6,
             say="Every decision traces to the rule and the circular behind it. Only then "
                 "does Gemma write, in her language, where to go and what to carry."),
        dict(kind="image", path=ASSETS / "model_choice.png",
             say="One finding worth the day. The smaller Gemma 4 marked a household "
                 "vehicle-free because someone stated their age. The cause was a rule I "
                 "had buried at the tail of an unrelated bullet. E4B applied it anyway. "
                 "E2B did not. The difference was not accuracy — it was how much slack "
                 "they leave for a wrong prompt."),
        dict(kind="title", big=["100% recall.  0% invention.", "49 tests. No cloud."],
             small="github.com/yunyiliu/kaggle-gemma-yojana-sarathi",
             say="Recall can drift. Invention is held at zero by code, not by the model "
                 "behaving."),
    ]


def main() -> None:
    if not DEMO.exists():
        sys.exit(f"missing {DEMO} - run: .venv/bin/python demo.py > {DEMO}")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir()

    scenes = build_scenes()
    parts = []
    for i, sc in enumerate(scenes):
        audio = BUILD / f"a{i:02d}.mp3"
        print(f"[{i + 1}/{len(scenes)}] narrating {sc['kind']}...")
        tts(sc["say"], audio)
        dur = max(duration(audio) + 0.6, 3.0)

        frames: list[tuple[pathlib.Path, float]] = []
        if sc["kind"] == "title":
            f = BUILD / f"f{i:02d}.png"
            title_frame(sc["big"], sc.get("small", ""), sc.get("deva", "")).save(f)
            frames = [(f, dur)]
        elif sc["kind"] == "image":
            f = BUILD / f"f{i:02d}.png"
            image_frame(sc["path"]).save(f)
            frames = [(f, dur)]
        else:
            lines = [l for l in sc["lines"]]
            step = sc.get("reveal", 4)
            stops = list(range(step, len(lines) + step, step)) or [len(lines)]
            per = dur / len(stops)
            for j, n in enumerate(stops):
                f = BUILD / f"f{i:02d}_{j:02d}.png"
                terminal_frame(lines[:n], sc["title"]).save(f)
                frames.append((f, per))

        concat = BUILD / f"c{i:02d}.txt"
        with concat.open("w") as fh:
            for f, d in frames:
                fh.write(f"file '{f.name}'\nduration {d:.3f}\n")
            fh.write(f"file '{frames[-1][0].name}'\n")

        part = BUILD / f"p{i:02d}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-i", str(audio),
            "-vf", f"fps={FPS},format=yuv420p,scale={W}:{H}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", "-t", f"{dur:.3f}", str(part),
        ], check=True, cwd=BUILD)
        parts.append(part)

    listing = BUILD / "parts.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    out = ROOT / "assets" / "demo.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-c", "copy", str(out),
    ], check=True, cwd=BUILD)
    print(f"\nwrote {out}  {out.stat().st_size / 1e6:.1f} MB  {duration(out):.0f}s")


if __name__ == "__main__":
    main()
