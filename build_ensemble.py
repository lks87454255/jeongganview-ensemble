#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ensemble.py
=================
AI-Hub 형식 원본(01.원천데이터 / 02.라벨링데이터)을 합주(Stem Mixer)용 에셋으로 변환합니다.

규칙(A안):
  ensemble/{project_name}/
  ├── {project_name}.json          ← 다운로드한 라벨 원본 그대로 복사 (수정 금지)
  ├── {project_name}.meta.json     ← 이 스크립트가 생성 (앱용 가공 정보)
  ├── {project_name}.m4a           ← master
  └── stems/{stem file_name}.m4a   ← 라벨의 file_name 에서 확장자만 .m4a

- 모든 파일/폴더명은 NFC(완성형)로 저장합니다. (macOS 다운로드 파일명은 NFD 인 경우가 많음)
- URL 인코딩은 generate_index.py / 앱과 동일: quote(nfc(part), safe="")
- 인코더: ffmpeg 가 있으면 ffmpeg, 없으면 macOS 내장 afconvert 사용.
  (모든 stem 을 같은 인코더로 변환하므로 AAC 인코더 지연이 동일 → 트랙 간 정렬 유지)

사용 예:
  python3 build_ensemble.py --source ~/Downloads/sample --song 0010_정악_풍류음악
  python3 build_ensemble.py --source ~/Downloads/sample --all
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from urllib.parse import quote

# ── 레포지토리 설정 ────────────────────────────────────────────
OWNER = "lks87454255"
REPO = "jeongganview-ensemble"
BRANCH = "main"

ENSEMBLE_DIR = "ensemble"
STEMS_DIR = "stems"
META_VERSION = 1

# 소스 검증 허용 오차: stem 간 길이 차이(ms)
DURATION_TOLERANCE_MS = 10

# ── 악기 매핑: instrumentSub → (key, group) ─────────────────────
#   key  : 앱 내부 식별자(ASCII, 저장 선택 상태 등에 사용)
#   group: UI 그룹(현악/관악/타악/성악/기타)
INSTRUMENTS = {
    "가야금": ("gayageum", "현악"),
    "25현가야금": ("gayageum25", "현악"),
    "거문고": ("geomungo", "현악"),
    "해금": ("haegeum", "현악"),
    "아쟁": ("ajaeng", "현악"),
    "양금": ("yanggeum", "현악"),
    "대금": ("daegeum", "관악"),
    "중금": ("junggeum", "관악"),
    "소금": ("sogeum", "관악"),
    "단소": ("danso", "관악"),
    "피리": ("piri", "관악"),
    "세피리": ("sepiri", "관악"),
    "향피리": ("hyangpiri", "관악"),
    "당피리": ("dangpiri", "관악"),
    "태평소": ("taepyeongso", "관악"),
    "생황": ("saenghwang", "관악"),
    "장구": ("janggu", "타악"),
    "북": ("buk", "타악"),
    "좌고": ("jwago", "타악"),
    "징": ("jing", "타악"),
    "꽹과리": ("kkwaenggwari", "타악"),
    "박": ("bak", "타악"),
    "편종": ("pyeonjong", "타악"),
    "편경": ("pyeongyeong", "타악"),
    "가곡": ("gagok", "성악"),
    "소리": ("sori", "성악"),
}


def nfc(s: str) -> str:
    """한글 조합형 → 완성형(NFC) 정규화 (generate_index.py / 앱과 동일)"""
    return unicodedata.normalize("NFC", s)


def raw_url(*parts: str) -> str:
    """raw.githubusercontent.com URL. Java URLEncoder.encode(..).replace("+","%20") 와 동일."""
    encoded = "/".join(quote(nfc(p), safe="") for p in parts)
    return f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{encoded}"


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"❌ {msg}")


# ── 소스 탐색 ──────────────────────────────────────────────────
def index_files(root: str) -> dict:
    """root 이하 모든 파일을 {nfc(파일명): 절대경로} 로 색인 (NFD/NFC 차이 흡수)."""
    result = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.startswith("."):
                continue
            key = nfc(f)
            if key in result:
                log(f"⚠️  같은 이름의 파일이 여러 개 있습니다. 첫 번째를 사용: {key}")
                continue
            result[key] = os.path.join(dirpath, f)
    return result


def find_label_jsons(label_root: str) -> dict:
    """02.라벨링데이터 이하의 {project_name}.json 을 {nfc(project_name): path} 로 반환."""
    found = {}
    for dirpath, _dirs, files in os.walk(label_root):
        for f in files:
            if f.lower().endswith(".json") and not f.startswith("."):
                found[nfc(os.path.splitext(f)[0])] = os.path.join(dirpath, f)
    return found


def find_song_source_dir(source_root: str, song: str) -> str:
    """01.원천데이터 이하에서 폴더명이 song 인 디렉터리를 찾는다."""
    for dirpath, dirs, _files in os.walk(source_root):
        for d in dirs:
            if nfc(d) == song:
                return os.path.join(dirpath, d)
    fail(f"원천데이터 폴더를 찾을 수 없습니다: {song}")
    return ""


# ── 오디오 도구 ────────────────────────────────────────────────
def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def probe(path: str) -> dict:
    """{sampleRate, channels, durationMs} 반환. ffprobe → afinfo 순서로 시도."""
    if which("ffprobe"):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels:format=duration",
             "-of", "json", path],
            capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        st = data["streams"][0]
        return {
            "sampleRate": int(st["sample_rate"]),
            "channels": int(st["channels"]),
            "durationMs": round(float(data["format"]["duration"]) * 1000),
        }
    if which("afinfo"):
        out = subprocess.run(["afinfo", path], capture_output=True, text=True, check=True).stdout
        fmt = re.search(r"Data format:\s+(\d+) ch,\s+(\d+) Hz", out)
        dur = re.search(r"estimated duration:\s+([\d.]+) sec", out)
        if not fmt or not dur:
            fail(f"afinfo 결과를 해석할 수 없습니다: {path}")
        return {
            "sampleRate": int(fmt.group(2)),
            "channels": int(fmt.group(1)),
            "durationMs": round(float(dur.group(1)) * 1000),
        }
    fail("ffprobe 또는 afinfo 가 필요합니다.")
    return {}


def encode_aac(src: str, dst: str, bitrate_k: int) -> None:
    """WAV → AAC(m4a). 임시 파일에 쓴 뒤 rename (중단 시 깨진 파일 방지)."""
    tmp = dst + ".tmp.m4a"
    if os.path.exists(tmp):
        os.remove(tmp)
    if which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vn",
               "-c:a", "aac", "-b:a", f"{bitrate_k}k", "-movflags", "+faststart", tmp]
    elif which("afconvert"):
        cmd = ["afconvert", "-f", "m4af", "-d", "aac", "-b", str(bitrate_k * 1000), src, tmp]
    else:
        fail("ffmpeg 또는 afconvert 가 필요합니다.")
        return
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        fail(f"인코딩 실패: {src}\n{e.stderr}")
    os.replace(tmp, dst)


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def needs_encode(src: str, dst: str, force: bool) -> bool:
    if force or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        return True
    return os.path.getmtime(src) > os.path.getmtime(dst)


# ── 곡 1개 처리 ────────────────────────────────────────────────
def build_song(song: str, label_path: str, source_root: str, out_root: str,
               bitrate_k: int, force: bool) -> dict:
    log(f"\n🎼 {song}")

    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    project = nfc(label.get("project_name", ""))
    if project != song:
        fail(f"라벨의 project_name({project}) 과 파일명({song}) 이 다릅니다.")

    master = label.get("master") or {}
    stems = label.get("stems") or []
    if not master.get("file_name"):
        fail("라벨에 master.file_name 이 없습니다.")
    if not stems:
        fail("라벨에 stems 가 없습니다.")

    files = index_files(find_song_source_dir(source_root, song))

    def src_of(file_name: str) -> str:
        p = files.get(nfc(file_name))
        if not p:
            fail(f"원천 음원을 찾을 수 없습니다: {file_name}")
        return p

    # ── 1) 소스 검증: 모든 stem 의 샘플레이트·길이 일치 ──────────
    master_src = src_of(master["file_name"])
    master_info = probe(master_src)
    stem_srcs = []
    for st in stems:
        p = src_of(st["file_name"])
        info = probe(p)
        if info["sampleRate"] != master_info["sampleRate"]:
            fail(f"샘플레이트 불일치: {st['file_name']} {info['sampleRate']} != {master_info['sampleRate']}")
        if abs(info["durationMs"] - master_info["durationMs"]) > DURATION_TOLERANCE_MS:
            fail(f"길이 불일치: {st['file_name']} {info['durationMs']}ms != {master_info['durationMs']}ms")
        stem_srcs.append((st, p, info))
    log(f"  ✅ 소스 검증: stem {len(stems)}개, {master_info['sampleRate']}Hz, "
        f"{master_info['channels']}ch, {master_info['durationMs']}ms")

    # ── 2) 출력 폴더 + 라벨 원본 복사 ────────────────────────────
    song_dir = os.path.join(out_root, ENSEMBLE_DIR, song)
    os.makedirs(os.path.join(song_dir, STEMS_DIR), exist_ok=True)
    label_name = f"{song}.json"
    shutil.copyfile(label_path, os.path.join(song_dir, label_name))  # 바이트 그대로 복사

    # ── 3) 인코딩 ──────────────────────────────────────────────
    def encode(src: str, rel_parts: list) -> dict:
        dst = os.path.join(song_dir, *rel_parts)
        if needs_encode(src, dst, force):
            log(f"  🔄 인코딩: {'/'.join(rel_parts)}")
            encode_aac(src, dst, bitrate_k)
        else:
            log(f"  ⏭️  최신 상태: {'/'.join(rel_parts)}")
        return {
            "url": raw_url(ENSEMBLE_DIR, song, *rel_parts),
            "sizeBytes": os.path.getsize(dst),
            "sha1": sha1_of(dst),
        }

    master_name = nfc(os.path.splitext(master["file_name"])[0]) + ".m4a"
    master_entry = {"fileName": master_name, **encode(master_src, [master_name])}

    used_keys = set()
    stem_entries = []
    for idx, (st, src, _info) in enumerate(stem_srcs):
        tags = st.get("tags") or {}
        name = nfc(tags.get("instrumentSub") or tags.get("instrumentMajor") or f"악기{idx + 1}")
        key, group = INSTRUMENTS.get(name, (None, "기타"))
        if key is None:
            key = f"stem{idx + 1:02d}"
            log(f"  ⚠️  매핑 없는 악기: '{name}' → key={key}, group=기타 (INSTRUMENTS 에 추가 권장)")
        base_key, n = key, 2
        while key in used_keys:  # 같은 악기가 여러 트랙인 경우 (예: 가야금 1, 2)
            key = f"{base_key}_{n}"
            n += 1
        used_keys.add(key)

        file_name = nfc(os.path.splitext(st["file_name"])[0]) + ".m4a"
        stem_entries.append({
            "key": key,
            "name": name,
            "group": group,
            "instrumentMajor": tags.get("instrumentMajor"),
            "sourceFileName": nfc(st["file_name"]),
            "fileName": file_name,
            **encode(src, [STEMS_DIR, file_name]),
            "defaultGain": 1.0,
            "pan": 0.0,
        })

    # 이전 빌드에서 남은 stem 파일 정리 (라벨에서 빠진 stem)
    expected = {e["fileName"] for e in stem_entries}
    for f in os.listdir(os.path.join(song_dir, STEMS_DIR)):
        if nfc(f) not in expected and not f.startswith("."):
            log(f"  🗑️  라벨에 없는 stem 삭제: {f}")
            os.remove(os.path.join(song_dir, STEMS_DIR, f))

    # ── 4) meta.json 생성 ──────────────────────────────────────
    mt = master.get("tags") or {}
    meta = {
        "version": META_VERSION,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "id": song,
        "title": mt.get("title"),
        "performers": mt.get("performers"),
        "genreMajor": mt.get("genreMajor"),
        "genreSub": mt.get("genreSub"),
        "genreDetail": mt.get("genreDetail"),
        "tempoBpm": mt.get("tempoBpm"),
        "timeSignature": mt.get("timeSignature"),
        "westernKey": mt.get("western_key"),
        "recordDate": mt.get("recordDate"),
        "moods": [m.get("mood") for m in (mt.get("moods") or []) if m.get("mood")],
        "sampleRate": master_info["sampleRate"],
        "channels": master_info["channels"],
        "durationMs": master_info["durationMs"],
        "codec": "aac",
        "bitrateKbps": bitrate_k,
        "labelUrl": raw_url(ENSEMBLE_DIR, song, label_name),
        "master": master_entry,
        "stems": stem_entries,
    }
    meta_path = os.path.join(song_dir, f"{song}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    total = master_entry["sizeBytes"] + sum(s["sizeBytes"] for s in stem_entries)
    log(f"  📝 {os.path.relpath(meta_path, out_root)}  (합계 {total / 1_000_000:.1f}MB)")
    return meta


# ── main ──────────────────────────────────────────────────────
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="AI-Hub 원본 → 합주용 에셋 변환")
    ap.add_argument("--source", required=True,
                    help="01.원천데이터 / 02.라벨링데이터 를 포함한 루트 폴더")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--song", action="append", help="project_name (여러 번 지정 가능)")
    g.add_argument("--all", action="store_true", help="라벨링데이터의 모든 곡 처리")
    ap.add_argument("--out", default=here, help="레포 루트 (기본: 스크립트 위치)")
    ap.add_argument("--bitrate", type=int, default=128, help="AAC 비트레이트 kbps (기본 128)")
    ap.add_argument("--force", action="store_true", help="기존 m4a 가 있어도 다시 인코딩")
    args = ap.parse_args()

    source = os.path.expanduser(args.source)
    label_root = next((os.path.join(source, d) for d in os.listdir(source)
                       if nfc(d) == "02.라벨링데이터"), None)
    raw_root = next((os.path.join(source, d) for d in os.listdir(source)
                     if nfc(d) == "01.원천데이터"), None)
    if not label_root or not raw_root:
        fail(f"{source} 아래에 01.원천데이터 / 02.라벨링데이터 폴더가 필요합니다.")

    labels = find_label_jsons(label_root)
    songs = sorted(labels) if args.all else [nfc(s) for s in args.song]
    if not songs:
        fail("처리할 곡이 없습니다.")

    log(f"인코더: {'ffmpeg' if which('ffmpeg') else 'afconvert'}  /  {args.bitrate}kbps  /  출력: {args.out}")
    for song in songs:
        if song not in labels:
            fail(f"라벨 JSON 을 찾을 수 없습니다: {song}.json")
        build_song(song, labels[song], raw_root, args.out, args.bitrate, args.force)

    log(f"\n✅ 완료: {len(songs)}곡. 다음 단계 → python3 generate_index.py 후 git add/commit/push")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
