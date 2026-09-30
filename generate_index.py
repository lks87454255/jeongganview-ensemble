#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_index.py
=================
ensemble/*/{project_name}.meta.json 을 스캔하여 곡 목록 ensemble-index.json 을 생성합니다.

- 앱은 raw.githubusercontent.com 에서 이 파일 1개만 받아 곡 목록을 표시하고,
  곡 선택 시 metaUrl 로 상세(meta.json)를 받습니다.
- 카테고리: "{genreMajor}/{genreSub}" (예: 정악/풍류음악)
- 내용이 바뀌지 않았으면 파일을 다시 쓰지 않습니다(generated 시각 유지)
  → 불필요한 git diff / ETag 변경 / 앱 재다운로드 방지.

사용:
  python3 build_ensemble.py --source ... --all   # 먼저 에셋 생성
  python3 generate_index.py
"""

import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from urllib.parse import quote

OWNER = "lks87454255"
REPO = "jeongganview-ensemble"
BRANCH = "main"

ENSEMBLE_DIR = "ensemble"
INDEX_FILE = "ensemble-index.json"
INDEX_VERSION = 1


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def raw_url(*parts: str) -> str:
    """Java URLEncoder.encode(..).replace("+","%20") 와 동일한 인코딩."""
    encoded = "/".join(quote(nfc(p), safe="") for p in parts)
    return f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{encoded}"


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_songs(root: str) -> list:
    base = os.path.join(root, ENSEMBLE_DIR)
    if not os.path.isdir(base):
        raise SystemExit(f"❌ {base} 폴더가 없습니다. build_ensemble.py 를 먼저 실행하세요.")

    songs = []
    for d in sorted(os.listdir(base), key=nfc):
        song_dir = os.path.join(base, d)
        if d.startswith(".") or not os.path.isdir(song_dir):
            continue
        song = nfc(d)
        meta_path = os.path.join(song_dir, f"{song}.meta.json")
        if not os.path.isfile(meta_path):
            print(f"⚠️  meta.json 없음, 건너뜀: {song}")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if nfc(meta.get("id", "")) != song:
            raise SystemExit(f"❌ meta.id({meta.get('id')}) 와 폴더명({song}) 이 다릅니다.")

        # 실제 파일 존재 확인 (meta 와 저장소 내용 불일치 방지)
        stems = meta.get("stems") or []
        missing = [s["fileName"] for s in stems
                   if not os.path.isfile(os.path.join(song_dir, "stems", s["fileName"]))]
        if missing:
            raise SystemExit(f"❌ {song}: stem 파일 누락 {missing}")

        master = meta.get("master") or {}
        total = (master.get("sizeBytes") or 0) + sum(s.get("sizeBytes") or 0 for s in stems)
        songs.append({
            "id": song,
            "title": meta.get("title"),
            "performers": meta.get("performers"),
            "genreMajor": meta.get("genreMajor"),
            "genreSub": meta.get("genreSub"),
            "genreDetail": meta.get("genreDetail"),
            "tempoBpm": meta.get("tempoBpm"),
            "durationMs": meta.get("durationMs"),
            "stemCount": len(stems),
            "instruments": [s.get("name") for s in stems],
            "totalBytes": total,
            "metaUrl": raw_url(ENSEMBLE_DIR, song, f"{song}.meta.json"),
            "metaSha1": sha1_of(meta_path),   # 앱: 값이 같으면 캐시된 meta 재사용
        })
    return songs


def build_index(songs: list) -> dict:
    cats = {}
    for s in songs:
        name = "/".join(p for p in (s.get("genreMajor"), s.get("genreSub")) if p) or "기타"
        cats.setdefault(name, []).append(s)
    return {
        "version": INDEX_VERSION,
        "owner": OWNER,
        "repo": REPO,
        "branch": BRANCH,
        "songCount": len(songs),
        "categories": [{"name": k, "songs": v} for k, v in sorted(cats.items())],
    }


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    index = build_index(load_songs(root))
    out_path = os.path.join(root, INDEX_FILE)

    # generated 를 제외한 내용이 같으면 기존 파일 유지
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        old_generated = old.pop("generated", None)
        if old == index:
            print(f"⏭️  변경 없음: {INDEX_FILE} (generated={old_generated})")
            return

    result = {"version": index["version"],
              "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              **{k: v for k, v in index.items() if k != "version"}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"✅ {INDEX_FILE}: {index['songCount']}곡, 카테고리 {len(index['categories'])}개")
    for c in index["categories"]:
        print(f"   - {c['name']}: {len(c['songs'])}곡")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
