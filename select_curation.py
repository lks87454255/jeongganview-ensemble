#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_curation.py — 1차 공개용 곡 목록(curation.txt) 자동 선정.

기준
  - 조각(60초 미만) 제외, 너무 긴 곡(기본 6분 초과) 제외
  - 장르별 목표 곡 수 안에서 장단·작품·악기가 겹치지 않게 고르게 선정
    · 판소리: 바탕(춘향가…) × 장단 골고루
    · 민요  : 지역 × 곡 골고루
    · 산조  : 악기(독주)별로 장단 골고루
    · 그 외 : 작품이 겹치지 않게
  - 이미 레포에 있는 곡은 항상 포함

사용:
  python3 select_curation.py --meta-csv 메타데이터.csv --zip-root "…/2.데이터(NIA)" > curation.txt
"""

import argparse
import collections
import os
import sys

from build_ensemble import ZipSource
from ensemble_common import ENSEMBLE_DIR, jangdan_of, load_meta_csv, nfc, work_movement

# 항상 포함: 96kHz 원본 곡 1개 (48kHz 변환 경로를 실제 에셋으로 확인)
EXTRA = {"0820_창작국악_창작국악": "창작국악 · 96kHz 원본"}

TARGET = {"판소리": 20, "민요": 14, "산조": 24, "풍류음악": 8, "대풍류": 4, "궁중음악": 4, "창작국악": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--zip-root", required=True)
    ap.add_argument("--min-sec", type=float, default=60)
    ap.add_argument("--max-sec", type=float, default=360)
    args = ap.parse_args()

    rows = load_meta_csv(os.path.expanduser(args.meta_csv))
    zs = ZipSource(args.zip_root)
    here = os.path.dirname(os.path.abspath(__file__))
    existing = {nfc(d) for d in os.listdir(os.path.join(here, ENSEMBLE_DIR))} \
        if os.path.isdir(os.path.join(here, ENSEMBLE_DIR)) else set()

    cands = collections.defaultdict(list)
    for song in zs.songs():
        r = rows.get(song)
        if not r:
            continue
        sec = float(r.get("오디오 길이(초)") or 0)
        if not (args.min_sec <= sec <= args.max_sec):
            continue
        g = nfc(r["genreSub"])
        work, mv = work_movement(r["title"], g, r["genreDetail"])
        jd, _ = jangdan_of(r["title"], g, r["timeSignature"], float(r["tempoBpm"] or 0))
        instr = [nfc(x.strip()) for x in (r.get("악기구성") or "").split(",") if x.strip()]
        lead = next((x for x in instr if x != "장구"), "") if g == "산조" else ""
        mb = os.path.getsize(zs.zips[song]) / 2 ** 20
        cands[g].append(dict(id=song, work=work, mv=mv, jd=jd or "-", lead=lead, sec=sec, mb=mb,
                             detail=nfc(r["genreDetail"]), title=nfc(r["title"])))

    chosen = []
    for g, n in TARGET.items():
        lst = sorted(cands.get(g, []), key=lambda c: (abs(c["sec"] - 180), c["id"]))  # 3분 안팎 우선
        if g == "판소리":
            group = lambda c: (c["detail"], c["jd"])
        elif g == "민요":
            group = lambda c: (c["detail"], c["mv"])
        elif g == "산조":
            group = lambda c: (c["lead"], c["jd"])
        else:
            group = lambda c: (c["work"],)
        buckets = collections.OrderedDict()
        for c in lst:
            buckets.setdefault(group(c), []).append(c)
        picked, round_ = [], 0
        while len(picked) < n and any(len(b) > round_ for b in buckets.values()):
            for b in buckets.values():   # 라운드로빈: 모든 묶음에서 하나씩
                if len(b) > round_ and len(picked) < n:
                    picked.append(b[round_])
            round_ += 1
        chosen += [(g, c) for c in picked]

    ids = {c["id"] for _g, c in chosen}
    print("# select_curation.py 자동 선정 — 한 줄에 project_name (# 뒤는 설명)")
    for song in sorted(existing - ids):
        print(f"{song}  # 기존 곡")
    for song, why in EXTRA.items():
        if song not in ids and song in zs.zips:
            print(f"{song}  # {why}")
    total = 0.0
    for g, c in chosen:
        total += c["mb"]
        print(f"{c['id']}  # {g} · {c['work']} / {c['mv'] or '-'} · {c['jd']} · {c['sec']:.0f}s")
    print(f"# {len(chosen)}곡, 원본 zip 합계 {total / 1024:.1f}GB", file=sys.stderr)


if __name__ == "__main__":
    main()
