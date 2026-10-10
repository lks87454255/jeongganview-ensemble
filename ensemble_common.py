#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_common.py — build_ensemble.py / generate_index.py / select_curation.py 공통 규칙.

- 파일명 NFC, URL 인코딩 (앱 EnsembleAssetRepository 와 동일)
- 악기 매핑 (이름 → key, 그룹), "대금2" 같은 번호 붙은 트랙 처리
- 메타데이터.csv(AI-Hub 71955 Other.zip) 파싱
- 카탈로그 파생값: 작품(work)/대목·악장(movement), 장단, 장단 한 바퀴 길이, 화면 모드

장단 한 바퀴 길이 규칙(실측 확인): cycleSec = timeSignature 분자 × 60 / tempoBpm
  예) 진양조 18/8 · 80BPM → 13.5초, 중모리 12/4 · 60BPM → 12초, 굿거리 12/8 · 240BPM → 3초
  (산조·판소리·민요·대풍류·창작 표본의 타악 트랙 자기상관 최대 지연과 일치)
"""

import csv
import os
import re
import unicodedata
from urllib.parse import quote

OWNER = "lks87454255"
REPO = "jeongganview-ensemble"
BRANCH = "main"
DEFAULT_BASE_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"

ENSEMBLE_DIR = "ensemble"
STEMS_DIR = "stems"
INDEX_DIR = "index"

TARGET_SAMPLE_RATE = 48000
FRAGMENT_MAX_MS = 60_000          # 이보다 짧으면 조각(fragment)
CYCLE_MIN_MS = 1_000              # 장단 한 바퀴로 인정할 길이 범위
CYCLE_MAX_MS = 40_000


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def base_url() -> str:
    """에셋 URL 접두사. 환경변수 ENSEMBLE_BASE_URL 로 바꿀 수 있다(예: CDN)."""
    return os.environ.get("ENSEMBLE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def asset_url(*parts: str) -> str:
    """Java URLEncoder.encode(..).replace("+","%20") 와 같은 경로 인코딩."""
    return base_url() + "/" + "/".join(quote(nfc(p), safe="") for p in parts)


# ── 악기 ──────────────────────────────────────────────────────
#   key  : 앱 내부 식별자(ASCII, 곡별 저장 상태에 사용)
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
    "소리북": ("soribuk", "타악"),
    "북": ("buk", "타악"),
    "대고": ("daego", "타악"),
    "좌고": ("jwago", "타악"),
    "징": ("jing", "타악"),
    "꽹과리": ("kkwaenggwari", "타악"),
    "바라": ("bara", "타악"),
    "박": ("bak", "타악"),
    "편종": ("pyeonjong", "타악"),
    "편경": ("pyeongyeong", "타악"),
    "방향": ("banghyang", "타악"),
    "목탁": ("moktak", "타악"),
    "방울": ("bangul", "타악"),
    "가곡": ("gagok", "성악"),
    "소리": ("sori", "성악"),
}

INSTRUMENT_MAJOR = {  # 라벨 JSON 의 instrumentMajor 와 같은 값 (CSV 로 라벨을 만들 때 사용)
    "가야금": "가야금", "거문고": "거문고", "해금": "해금", "아쟁": "아쟁", "양금": "양금",
    "대금": "대금", "피리": "피리",
}


def split_instrument(name: str):
    """'대금2' → ('대금', 2), '피리1' → ('피리', 1), '대금' → ('대금', 1). 매핑에 있는 이름은 그대로."""
    name = nfc(name).strip()
    if name in INSTRUMENTS:
        return name, 1
    m = re.match(r"^(.*?)(\d+)$", name)
    if m and m.group(1) in INSTRUMENTS:
        return m.group(1), int(m.group(2))
    return name, 1


def instrument_info(name: str):
    """이름 → (baseName, key or None, group)."""
    base, _n = split_instrument(name)
    key, group = INSTRUMENTS.get(base, (None, "기타"))
    return base, key, group


# ── 메타데이터.csv ────────────────────────────────────────────
def parse_moods(s: str):
    """'의례적인(2) | 장엄한(2)' → [{'mood':'의례적인','count':2}, ...]"""
    out = []
    for part in (s or "").split("|"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\s*\((\d+)\)$", part)
        out.append({"mood": m.group(1).strip(), "count": int(m.group(2))} if m else {"mood": part, "count": 1})
    return out


def parse_date(s: str):
    """'2025. 8. 5' → '2025-08-05' (형식이 다르면 원문)."""
    m = re.match(r"^\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", s or "")
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else (s or None)


def _num(s):
    try:
        v = float(s)
        return int(v) if v.is_integer() else v
    except (TypeError, ValueError):
        return None


def load_meta_csv(path: str) -> dict:
    """메타데이터.csv → {project_name(NFC): row dict}"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {nfc(r["project_name"]): r for r in csv.DictReader(f)}


def label_from_csv(row: dict, stem_names: list) -> dict:
    """라벨 JSON 이 없는 곡: CSV 행 + 실제 스템 파일 이름으로 AI-Hub 라벨과 같은 형식을 만든다."""
    project = nfc(row["project_name"])
    stems = []
    for s in stem_names:
        base, _key, group = instrument_info(s)
        major = INSTRUMENT_MAJOR.get(base) or ("타악기" if group == "타악" else "기타")
        stems.append({"file_name": f"{project}_{s}.wav",
                      "tags": {"instrumentMajor": major, "instrumentSub": s}})
    return {
        "project_name": project,
        "_source": "메타데이터.csv",
        "master": {
            "file_name": f"{project}.wav",
            "tags": {
                "sampleRate": TARGET_SAMPLE_RATE,
                "bitDepth": 24,
                "performers": "국립국악원",
                "genreMajor": nfc(row.get("genreMajor", "")) or None,
                "genreSub": nfc(row.get("genreSub", "")) or None,
                "genreDetail": nfc(row.get("genreDetail", "")) or None,
                "title": nfc(row.get("title", "")) or None,
                "tempoBpm": _num(row.get("tempoBpm")),
                "timeSignature": (row.get("timeSignature") or "").strip() or None,
                "recordDate": parse_date(row.get("recordDate")),
                "moods": parse_moods(row.get("mood_count")),
                "western_key": (row.get("western_key") or "").strip() or None,
            },
        },
        "stems": stems,
    }


# ── 파생값 ────────────────────────────────────────────────────
# 제목에서 찾는 장단 이름. '타령'·'도드리'는 곡 이름(까투리타령, 계면가락도드리)에 흔해 제외한다.
JANGDAN_NAMES = ["진양조", "엇중모리", "중중모리", "중모리", "자진모리", "휘모리", "단모리",
                 "굿거리", "엇모리", "세마치"]
_JANGDAN_RE = re.compile("(" + "|".join(sorted(JANGDAN_NAMES, key=len, reverse=True)) + ")")


def mode_of(genre_sub: str) -> str:
    """화면 모드: 판소리·민요 = 반주(소리 없음), 산조 = 독주, 나머지 = 합주."""
    g = nfc(genre_sub or "")
    if g in ("판소리", "민요"):
        return "accompaniment"
    if g == "산조":
        return "solo"
    return "ensemble"


def time_sig_beats(ts: str):
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", ts or "")
    return int(m.group(1)) if m else None


def cycle_ms(time_signature: str, tempo_bpm) -> int:
    """장단 한 바퀴 길이(ms). 범위를 벗어나면(제례악 1/4 등) None."""
    beats = time_sig_beats(time_signature)
    if not beats or beats < 2 or not tempo_bpm or tempo_bpm <= 0:
        return None
    ms = round(beats * 60_000 / float(tempo_bpm))
    return ms if CYCLE_MIN_MS <= ms <= CYCLE_MAX_MS else None


def jangdan_of(title: str, genre_sub: str, time_signature: str, tempo_bpm):
    """(장단 이름, 추정 여부). 제목에 있으면 그대로, 판소리·민요는 박자·빠르기로 추정."""
    m = _JANGDAN_RE.search(nfc(title or ""))
    if m:
        return m.group(1), False
    g = nfc(genre_sub or "")
    ts = (time_signature or "").replace(" ", "")
    bpm = float(tempo_bpm) if tempo_bpm else 0.0
    guess = None
    if g == "판소리":
        if ts == "18/8":
            guess = "진양조"
        elif ts == "12/4":
            guess = "중모리"
        elif ts == "10/8":
            guess = "엇모리"
        elif ts == "4/4":
            guess = "휘모리"
        elif ts == "12/8":
            guess = "중중모리" if 0 < bpm <= 170 else ("자진모리" if bpm >= 220 else None)
    elif g == "민요":
        if ts == "9/8":
            guess = "세마치"
        elif ts == "10/8":
            guess = "엇모리"
        elif ts == "12/4":
            guess = "중모리"
    elif g == "산조":
        guess = {"18/8": "진양조", "12/4": "중모리", "10/8": "엇모리", "4/4": "휘모리"}.get(ts)
    return guess, guess is not None


def _strip_number(s: str):
    """끝의 '3', '(1)', '2장', '1장~3장', '2악장' → (앞부분, 번호부분 or None)"""
    s = s.strip()
    m = re.match(r"^(.*?)[\s]*((?:\(?\d+\)?)|(?:\d+장(?:\s*~\s*\d+장)?)|(?:\d+악장))$", s)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    return s, None


def work_movement(title: str, genre_sub: str, genre_detail: str):
    """(작품, 대목/악장).
    판소리: (바탕=genreDetail, 대목=title), 민요: (지역=genreDetail, 곡=title),
    그 외: '작품 중 악장' 또는 끝 번호를 떼어 작품으로 묶는다."""
    t = nfc(title or "").strip()
    g = nfc(genre_sub or "")
    d = nfc(genre_detail or "").strip()
    if g in ("판소리", "민요") and d:
        return d, t or None
    if " 중 " in t:
        w, mv = t.split(" 중 ", 1)
        return w.strip().rstrip(" -–·,"), mv.strip() or None
    w, num = _strip_number(t)
    w = w.rstrip(" -–·,") or t
    return w, num


def natural_key(s: str):
    """'중모리 2' < '중모리 10' 정렬용"""
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", nfc(s or ""))]


def load_excluded(root: str) -> dict:
    """excluded.txt → {project_name: 이유}"""
    path = os.path.join(root, "excluded.txt")
    out = {}
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            body, _, why = line.partition("#")
            body = body.strip()
            if body:
                out[nfc(body.split()[0])] = why.strip()
    return out

