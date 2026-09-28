from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# These are the two example images used in the assignment demo.
SOURCES = {
    DATA / "rgb" / "rgb_human.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/3c/Man_Standing.jpg",
    DATA / "thermal" / "thermal_human.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b4/Man_in_water_-_IR_image.jpg",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())


for path, url in SOURCES.items():
    if path.exists():
        print(f"Already exists: {path}")
        continue

    print(f"Downloading: {url}")
    download(url, path)
    print(f"Saved: {path}")

print("Done.")
