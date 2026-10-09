#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_index.py (v2)
======================
ensemble/*/{곡}.meta.json 을 스캔해 곡 목록을 만든다.

  ensemble-index.json          ← 앱이 처음 받는 파일 (작음)
    version: 2
    genres[]: {key, name, genreMajor, mode, songCount, totalBytes, url, sha1}
    categories[]: v1 앱 호환용 목록 (v1 필드만)
  index/v2/{genreKey}.json     ← 장르별 곡 요약 (작품/대목/장단/모드 등)

- 앱(v2)은 ensemble-index.json → 장르 파일(sha1 이 같으면 캐시 사용)을 읽는다.
- 내용이 바뀌지 않았으면 파일을 다시 쓰지 않는다(generated 유지 → git diff / ETag 안정).
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from ensemble_common import (
    ENSEMBLE_DIR, FRAGMENT_MAX_MS, INDEX_DIR, OWNER, REPO, BRANCH,
    asset_url, cycle_ms, jangdan_of, mode_of, natural_key, nfc, work_movement,
)

INDEX_FILE = "ensemble-index.json"
INDEX_VERSION = 2
GENRE_INDEX_SUBDIR = "v2"

# 장르 표시 순서와 파일 키 (ASCII)
GENRES = [
    ("판소리", "pansori"),
    ("민요", "minyo"),
    ("산조", "sanjo"),
    ("풍류음악", "pungryu"),
    ("대풍류", "daepungryu"),
    ("궁중음악", "gungjung"),
    ("창작국악", "changjak"),
]
GENRE_KEY = dict(GENRES)
GENRE_ORDER = {name: i for i, (name, _k) in enumerate(GENRES)}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha1_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def write_if_changed(path: str, data: dict) -> bool:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            old.pop("generated", None)
            if old == {k: v for k, v in data.items() if k != "generated"}:
                return False
        except (OSError, ValueError):
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return True


def summarize(meta: dict, meta_path: str, song_dir: str) -> dict:
    song = nfc(meta["id"])
    stems = meta.get("stems") or []
    missing = [s["fileName"] for s in stems
               if not os.path.isfile(os.path.join(song_dir, "stems", s["fileName"]))]
    if missing:
        raise SystemExit(f"❌ {song}: stem 파일 누락 {missing}")
    master = meta.get("master") or {}
    total = (master.get("sizeBytes") or 0) + sum(s.get("sizeBytes") or 0 for s in stems)

    # v1 meta(이전 빌드)도 받아들이도록 빠진 파생값은 여기서 계산
    title, gsub, gdet = meta.get("title"), meta.get("genreSub"), meta.get("genreDetail")
    ts, bpm = meta.get("timeSignature"), meta.get("tempoBpm")
    work, movement = (meta["work"], meta.get("movement")) if meta.get("work") else work_movement(title, gsub, gdet)
    if "jangdan" in meta:
        jangdan, inferred = meta.get("jangdan"), bool(meta.get("jangdanInferred"))
    else:
        jangdan, inferred = jangdan_of(title, gsub, ts, bpm)
    mode = meta.get("mode") or mode_of(gsub)
    lead = meta.get("leadKey")
    lead_name = next((s.get("baseName") or s.get("name") for s in stems if s.get("key") == lead), None)
    duration = meta.get("durationMs") or 0
    return {
        "id": song,
        "title": title,
        "work": work,
        "movement": movement,
        "performers": meta.get("performers"),
        "genreMajor": meta.get("genreMajor"),
        "genreSub": gsub,
        "genreDetail": gdet,
        "tempoBpm": bpm,
        "timeSignature": ts,
        "jangdan": jangdan,
        "jangdanInferred": inferred,
        "cycleMs": meta.get("cycleMs") if "cycleMs" in meta else cycle_ms(ts, bpm),
        "mode": mode,
        "leadInstrument": lead_name,
        "fragment": bool(meta.get("fragment", duration < FRAGMENT_MAX_MS)),
        "westernKey": meta.get("westernKey"),
        "moods": meta.get("moods") or [],
        "durationMs": duration,
        "stemCount": len(stems),
        "instruments": [s.get("name") for s in stems],
        "totalBytes": total,
        "metaUrl": asset_url(ENSEMBLE_DIR, song, f"{song}.meta.json"),
        "metaSha1": sha1_of(meta_path),
    }


V1_FIELDS = ("id", "title", "performers", "genreMajor", "genreSub", "genreDetail", "tempoBpm",
             "durationMs", "stemCount", "instruments", "totalBytes", "metaUrl", "metaSha1")


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
        songs.append(summarize(meta, meta_path, song_dir))
    return songs


def song_sort_key(s: dict):
    return (natural_key(s.get("work") or ""), natural_key(s.get("movement") or ""), s["id"])


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    songs = load_songs(root)

    by_genre = {}
    for s in songs:
        by_genre.setdefault(nfc(s.get("genreSub") or "기타"), []).append(s)

    genres = []
    expected_files = set()
    for name in sorted(by_genre, key=lambda n: (GENRE_ORDER.get(n, len(GENRES)), n)):
        lst = sorted(by_genre[name], key=song_sort_key)
        key = GENRE_KEY.get(name) or ("g" + hashlib.sha1(name.encode()).hexdigest()[:8])
        rel = [INDEX_DIR, GENRE_INDEX_SUBDIR, f"{key}.json"]
        path = os.path.join(root, *rel)
        expected_files.add(os.path.abspath(path))
        data = {"version": INDEX_VERSION, "generated": now(), "key": key, "name": name,
                "genreMajor": lst[0].get("genreMajor"), "songCount": len(lst), "songs": lst}
        changed = write_if_changed(path, data)
        genres.append({
            "key": key,
            "name": name,
            "genreMajor": lst[0].get("genreMajor"),
            "mode": mode_of(name),
            "songCount": len(lst),
            "totalBytes": sum(s["totalBytes"] for s in lst),
            "url": asset_url(*rel),
            "sha1": sha1_of(path),
        })
        print(f"   {'✏️ ' if changed else '⏭️ '} {name}: {len(lst)}곡 → {'/'.join(rel)}")

    # 더 이상 없는 장르 파일 정리
    gdir = os.path.join(root, INDEX_DIR, GENRE_INDEX_SUBDIR)
    if os.path.isdir(gdir):
        for f in os.listdir(gdir):
            p = os.path.abspath(os.path.join(gdir, f))
            if f.endswith(".json") and p not in expected_files:
                print(f"   🗑️  {f}")
                os.remove(p)

    # v1 앱 호환 categories ("{genreMajor}/{genreSub}")
    cats = {}
    for s in sorted(songs, key=lambda s: (GENRE_ORDER.get(s.get("genreSub"), 99), song_sort_key(s))):
        name = "/".join(p for p in (s.get("genreMajor"), s.get("genreSub")) if p) or "기타"
        cats.setdefault(name, []).append({k: s.get(k) for k in V1_FIELDS})

    index = {
        "version": INDEX_VERSION,
        "generated": now(),
        "owner": OWNER,
        "repo": REPO,
        "branch": BRANCH,
        "songCount": len(songs),
        "totalBytes": sum(s["totalBytes"] for s in songs),
        "genres": genres,
        "categories": [{"name": k, "songs": v} for k, v in cats.items()],
    }
    out = os.path.join(root, INDEX_FILE)
    if write_if_changed(out, index):
        print(f"✅ {INDEX_FILE}: {len(songs)}곡, 장르 {len(genres)}개, "
              f"합계 {index['totalBytes'] / 1e9:.2f}GB")
    else:
        print(f"⏭️  변경 없음: {INDEX_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
