"""Download a small CC0 Poly Haven asset set into an ignored local cache."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ASSETS = {
    "sofa_03": "model",
    "modern_arm_chair_01": "model",
    "potted_plant_02": "model",
    "dining_chair_02": "model",
    "wood_floor": "texture",
    "denim_fabric": "texture",
    "white_plaster_02": "texture",
}


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "RoomGraph/0.1 (research visualization)"}
        ),
        timeout=90,
    )


def main():
    root = Path("assets/cache")
    root.mkdir(parents=True, exist_ok=True)
    catalog = {}
    jobs = []
    for name, kind in ASSETS.items():
        with request(f"https://api.polyhaven.com/files/{name}") as response:
            files = json.load(response)
        if kind == "model":
            gltf = files["gltf"]["1k"]["gltf"]
            selected = {f"{name}.gltf": gltf, **gltf["include"]}
        else:
            selected = {
                f"{key}.jpg": files[key]["1k"]["jpg"] for key in ("Diffuse", "nor_gl", "Rough")
            }
        catalog[name] = {
            "source": f"https://polyhaven.com/a/{name}",
            "license": "CC0-1.0",
            "kind": kind,
            "files": {},
        }
        for relative, details in selected.items():
            catalog[name]["files"][relative] = {k: details[k] for k in ("url", "md5", "size")}
            jobs.append((root / name / relative, details))

    def download(job):
        path, details = job
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != details["md5"]:
            with request(details["url"]) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != details["md5"]:
                raise RuntimeError(f"Checksum mismatch: {path}")
            path.write_bytes(payload)
        return str(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for path in pool.map(download, jobs):
            print(path, flush=True)
    Path("assets/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
