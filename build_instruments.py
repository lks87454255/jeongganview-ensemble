#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_instruments.py
====================
소리 마당 stem(원본 WAV)에서 "음별 단음 샘플"을 잘라 조율 / 정간보 만들기 화면용 샘플 악기를 만듭니다.

출력(레포 루트 기준):
  instruments/{key}/pack.json          ← 앱이 읽는 샘플 목록 (URL/크기/sha1/루프 지점)
  instruments/{key}/{key}_{midi}.wav   ← 48kHz mono PCM16, 음높이를 A4=440 평균율에 정확히 보정

처리 순서(음마다):
  1) YIN 으로 음높이 추적 → 녹음 전체의 조율 치우침(cent) 측정
  2) 같은 음이 흔들림 없이(농현/시김새 제외) 이어지는 "안정 구간"을 찾고, 곡 전체에서 가장 좋은 구간 선택
  3) 해당 구간의 실측 음높이 → 목표 주파수(A4=440)로 리샘플링해 정확히 보정 (조율 기준음 용도)
  4) 지속음 악기(sustain)는 안정 구간 안에 정수 주기 루프를 잡고 크로스페이드 → 누르는 동안 끊김 없이 유지
  5) 음량 정규화(RMS), 시작 페이드인

사용:
  python3 build_instruments.py --source ~/Downloads/sample --inst daegeum
  python3 build_instruments.py --source ~/Downloads/sample --inst daegeum --preview /tmp/preview
"""

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from urllib.parse import quote

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

OWNER, REPO, BRANCH = "lks87454255", "jeongganview-ensemble", "main"
OUT_SR = 48000
TARGET_RMS_DB = -20.0
PACK_VERSION = 1

# key → 설정
#   name   : 라벨 JSON instrumentSub
#   fmin/fmax : 음높이 탐색 범위(Hz)
#   sustain: 지속음(루프 필요) 여부
INSTRUMENTS = {
    "daegeum": {"name": "대금", "fmin": 180.0, "fmax": 1500.0, "sustain": True},
    "haegeum": {"name": "해금", "fmin": 250.0, "fmax": 2000.0, "sustain": True},
    "piri": {"name": "피리", "fmin": 180.0, "fmax": 1600.0, "sustain": True},
    "danso": {"name": "단소", "fmin": 350.0, "fmax": 2000.0, "sustain": True},
    "gayageum": {"name": "가야금", "fmin": 80.0, "fmax": 800.0, "sustain": False},
    "geomungo": {"name": "거문고", "fmin": 60.0, "fmax": 500.0, "sustain": False},
    "yanggeum": {"name": "양금", "fmin": 120.0, "fmax": 1000.0, "sustain": False},
}

HOP = 240            # 5ms @48k
WIN = 2048
YIN_THRESHOLD = 0.15
MIN_RMS_DB = -45.0   # 이보다 작으면 무음/번짐으로 간주
NOTE_TOL_CENT = 40   # 같은 음으로 볼 편차
STABLE_TOL_CENT = 12 # 안정 구간 편차
MIN_CORE_SEC = 0.45  # 안정 구간 최소 길이
MAX_SAMPLE_SEC = 2.5
ATTACK_PAD_SEC = 0.02
XFADE_SEC = 0.04
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def nfc(s): return unicodedata.normalize("NFC", s)
def note_name(m): return f"{NAMES[m % 12]}{m // 12 - 1}"
def midi_hz(m): return 440.0 * 2 ** ((m - 69) / 12)
def hz_midi(f): return 69 + 12 * np.log2(f / 440.0)
def raw_url(*parts): return f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/" + "/".join(quote(nfc(p), safe="") for p in parts)


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_mono(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != OUT_SR:
        x = resample(x, int(round(len(x) * OUT_SR / sr)))
    return x


# ── YIN (프레임 일괄, FFT 기반 차분 함수) ─────────────────────────
def yin_track(x, sr, fmin, fmax):
    max_lag = int(sr / fmin) + 2
    min_lag = max(2, int(sr / fmax))
    n_frames = max(0, (len(x) - WIN - max_lag) // HOP)
    f0 = np.full(n_frames, np.nan)
    rms_db = np.full(n_frames, -120.0)
    nfft = 1 << int(np.ceil(np.log2(WIN + max_lag)))
    csum = np.concatenate([[0.0], np.cumsum(x * x)])
    batch = 512
    for b0 in range(0, n_frames, batch):
        idx = np.arange(b0, min(n_frames, b0 + batch))
        starts = idx * HOP
        seg = np.stack([x[s:s + WIN + max_lag] for s in starts])
        head = seg[:, :WIN]
        r = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(head, nfft)), nfft)[:, :max_lag]
        e0 = csum[starts + WIN] - csum[starts]
        lags = np.arange(max_lag)
        etau = csum[starts[:, None] + lags + WIN] - csum[starts[:, None] + lags]
        d = e0[:, None] + etau - 2 * r
        d[:, 0] = 0
        cm = np.cumsum(d[:, 1:], axis=1) / np.arange(1, max_lag)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] / np.maximum(cm, 1e-12)
        rms_db[idx] = 10 * np.log10(e0 / WIN + 1e-12)
        for k, row in enumerate(cmnd):
            below = np.where(row[min_lag:] < YIN_THRESHOLD)[0]
            if len(below) == 0:
                continue
            t = below[0] + min_lag
            while t + 1 < max_lag and row[t + 1] < row[t]:
                t += 1
            if 1 <= t < max_lag - 1:  # 포물선 보간
                a, bb, c = row[t - 1], row[t], row[t + 1]
                den = a - 2 * bb + c
                t = t + (0.5 * (a - c) / den if den != 0 else 0.0)
            f0[idx[k]] = sr / t
    f0[rms_db < MIN_RMS_DB] = np.nan
    return f0, rms_db


# ── 안정 구간 탐색 ──────────────────────────────────────────────
def find_segments(f0, rms_db, offset_cent):
    """(midi, core_start_frame, core_end_frame, median_hz, std_cent, rms_db) 목록"""
    midi_f = hz_midi(f0) - offset_cent / 100.0
    note = np.round(midi_f)
    dev = (midi_f - note) * 100
    ok = np.isfinite(midi_f) & (np.abs(dev) < NOTE_TOL_CENT)
    segs, i, n = [], 0, len(f0)
    min_frames = int(MIN_CORE_SEC * OUT_SR / HOP)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and note[j + 1] == note[i]:
            j += 1
        if j - i + 1 >= min_frames:
            cents = hz_midi(f0[i:j + 1]) * 100
            med = np.median(cents)
            stable = np.abs(cents - med) < STABLE_TOL_CENT
            # 가장 긴 연속 안정 구간
            best, run_s = (0, 0), None
            for k, s in enumerate(np.append(stable, False)):
                if s and run_s is None:
                    run_s = k
                elif not s and run_s is not None:
                    if k - run_s > best[1] - best[0]:
                        best = (run_s, k)
                    run_s = None
            cs, ce = i + best[0], i + best[1]
            if ce - cs >= min_frames:
                core_c = hz_midi(f0[cs:ce]) * 100
                segs.append(dict(
                    midi=int(note[i]), seg_start=i, core_start=cs, core_end=ce,
                    hz=float(440 * 2 ** ((np.median(core_c) / 100 - 69) / 12)),
                    std=float(np.std(core_c)), rms=float(np.mean(rms_db[cs:ce])),
                ))
        i = j + 1
    return segs


def score(s):
    dur = (s["core_end"] - s["core_start"]) * HOP / OUT_SR
    return min(dur, 1.5) * 2 + (s["rms"] + 40) / 10 - s["std"] / 4


# ── 샘플 제작 ──────────────────────────────────────────────────
def make_sample(x, s, sustain):
    target = midi_hz(s["midi"])
    ratio = target / s["hz"]
    a = max(0, s["seg_start"] * HOP - int(ATTACK_PAD_SEC * OUT_SR))
    core_s = s["core_start"] * HOP + WIN // 2
    core_e = s["core_end"] * HOP + WIN // 2
    b = min(len(x), core_e, a + int(MAX_SAMPLE_SEC * OUT_SR * ratio))
    raw = x[a:b]
    y = resample(raw, max(8, int(round(len(raw) / ratio))))  # 길이 1/ratio → 음높이 ×ratio
    cs = int((core_s - a) / ratio)
    ce = min(len(y), int((core_e - a) / ratio))

    # 음량 정규화 (안정 구간 RMS 기준)
    ref = y[cs:ce] if ce - cs > 1000 else y
    g = 10 ** (TARGET_RMS_DB / 20) / (np.sqrt(np.mean(ref ** 2)) + 1e-12)
    y = y * g
    peak = np.max(np.abs(y))
    if peak > 0.95:
        y *= 0.95 / peak

    fade_in = min(len(y), int(0.005 * OUT_SR))
    y[:fade_in] *= np.linspace(0, 1, fade_in)

    loop = None
    if sustain:
        period = OUT_SR / target
        xf = int(XFADE_SEC * OUT_SR)
        loop_end = ce - int(0.03 * OUT_SR)
        want = min(0.5, (loop_end - cs) / OUT_SR * 0.6)
        n_per = max(4, int(round(want * OUT_SR / period)))
        base_start = int(round(loop_end - n_per * period))
        # 이음새 상관이 가장 높은 시작점 탐색 (±1주기)
        best_c, best_s = -2, base_start
        w = int(max(period * 2, 256))
        for st in range(base_start - int(period), base_start + int(period) + 1):
            if st - xf < cs or st + w > loop_end:
                continue
            u, v = y[st:st + w], y[loop_end:loop_end + w] if loop_end + w <= len(y) else None
            if v is None or len(v) < w:
                u, v = y[st - w:st], y[loop_end - w:loop_end]
            c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
            if c > best_c:
                best_c, best_s = c, st
        loop_start = best_s
        if loop_start - xf < 0 or loop_end - loop_start < xf * 2:
            raise ValueError("루프 구간 부족")
        # 루프 구간 음량 평탄화: 자연 감쇠 때문에 루프마다 음량이 '울렁'이는 것을 방지.
        #   진입점(loop_start - xf)의 음량을 목표로, 그 이후 구간 이득을 매끄럽게 맞춘다(진입점 이득=1 → 단차 없음).
        env_w = int(max(4 * period, 0.03 * OUT_SR))
        k = np.ones(env_w) / env_w
        env = np.sqrt(np.convolve(y ** 2, k, mode="same")) + 1e-9
        a0 = loop_start - xf
        tgt = env[a0]
        gain = np.clip(tgt / env[a0:loop_end], 0.5, 2.0)
        gain = np.convolve(np.pad(gain, (env_w, env_w), mode="edge"), k, mode="same")[env_w:-env_w]
        gain = gain / gain[0]
        y[a0:loop_end] *= gain
        # 크로스페이드: loop_end 직전 xf 구간을 loop_start 직전 구간과 섞음.
        #   두 구간은 주기를 맞춰 상관이 높으므로(>0.9) 등이득(선형) 페이드가 음량 불룩함이 없다.
        t = np.linspace(0, 1, xf)
        fo, fi = (1 - t, t) if best_c > 0.9 else (np.cos(t * np.pi / 2), np.sin(t * np.pi / 2))
        y[loop_end - xf:loop_end] = y[loop_end - xf:loop_end] * fo + y[loop_start - xf:loop_start] * fi
        y = y[:loop_end]
        loop = (int(loop_start), int(loop_end), float(best_c))
    else:
        fo = min(len(y), int(0.03 * OUT_SR))
        y[-fo:] *= np.linspace(1, 0, fo)
    return y, loop


def write_wav(path, y):
    pcm = np.clip(np.round(y * 32767), -32768, 32767).astype("<i2")
    wavfile.write(path, OUT_SR, pcm)


def render_preview(y, loop, seconds=3.0):
    """루프를 포함해 seconds 길이로 이어 재생한 신호 (검증/청음용)"""
    if loop is None:
        return y
    ls, le, _ = loop
    out = list(y)
    while len(out) < seconds * OUT_SR:
        out.extend(y[ls:le])
    return np.array(out[:int(seconds * OUT_SR)])


# ── main ──────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--inst", required=True, choices=sorted(INSTRUMENTS))
    ap.add_argument("--out", default=here)
    ap.add_argument("--preview", help="루프 이어붙인 3초 미리듣기 WAV 출력 폴더 (레포 밖 권장)")
    args = ap.parse_args()

    cfg = INSTRUMENTS[args.inst]
    src_root = os.path.expanduser(args.source)
    raw_root = next(os.path.join(src_root, d) for d in os.listdir(src_root) if nfc(d) == "01.원천데이터")
    stems = []
    for dp, _ds, fs in os.walk(raw_root):
        for f in fs:
            if nfc(f).endswith(f"_{cfg['name']}.wav"):
                stems.append(os.path.join(dp, f))
    if not stems:
        raise SystemExit(f"❌ '{cfg['name']}' stem 이 없습니다")

    candidates = []
    for p in sorted(stems):
        song = nfc(os.path.basename(p))[: -len(f"_{cfg['name']}.wav")]
        print(f"🎼 {song}: 분석 중…", flush=True)
        x = load_mono(p)
        f0, rms = yin_track(x, OUT_SR, cfg["fmin"], cfg["fmax"])
        voiced = f0[np.isfinite(f0)]
        dev = (hz_midi(voiced) - np.round(hz_midi(voiced))) * 100
        offset = float(np.median(dev))
        segs = find_segments(f0, rms, offset)
        print(f"   조율 치우침 {offset:+.1f} cent, 유성 프레임 {len(voiced)}, 안정 구간 {len(segs)}개")
        for s in segs:
            candidates.append((score(s), song, x, s))

    best = {}
    for sc, song, x, s in candidates:
        if s["midi"] not in best or sc > best[s["midi"]][0]:
            best[s["midi"]] = (sc, song, x, s)

    out_dir = os.path.join(args.out, "instruments", args.inst)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".wav"):
            os.remove(os.path.join(out_dir, f))
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    samples = []
    for midi in sorted(best):
        _sc, song, x, s = best[midi]
        try:
            y, loop = make_sample(x, s, cfg["sustain"])
        except ValueError as e:
            print(f"   ⚠️  {note_name(midi)}: 건너뜀 ({e})")
            continue
        fname = f"{args.inst}_{midi}.wav"
        path = os.path.join(out_dir, fname)
        write_wav(path, y)
        # 검증: 결과 음높이 재측정
        f0v, _ = yin_track(np.concatenate([y, np.zeros(WIN * 2)]), OUT_SR, cfg["fmin"], cfg["fmax"])
        fv = f0v[np.isfinite(f0v)]
        err = float(np.median(hz_midi(fv) * 100 - midi * 100)) if len(fv) else float("nan")
        entry = {
            "midi": midi, "note": note_name(midi), "fileName": fname,
            "url": raw_url("instruments", args.inst, fname),
            "sizeBytes": os.path.getsize(path), "sha1": sha1_of(path),
            "rootHz": round(midi_hz(midi), 4), "frames": len(y),
            "sourceSong": song, "sourceHz": round(s["hz"], 2), "correctionCent": round(1200 * np.log2(midi_hz(midi) / s["hz"]), 1),
            "pitchStdCent": round(s["std"], 1), "verifyCent": round(err, 1),
        }
        if loop:
            entry.update(loopStart=loop[0], loopEnd=loop[1], loopCorr=round(loop[2], 3))
        samples.append(entry)
        lp = f" loop={loop[0]}..{loop[1]} corr={loop[2]:.3f}" if loop else ""
        print(f"   ✅ {note_name(midi):4s} 보정 {entry['correctionCent']:+6.1f}c → 재측정 {err:+5.1f}c  "
              f"흔들림 {s['std']:.1f}c  길이 {len(y)/OUT_SR:.2f}s{lp}")
        if args.preview:
            write_wav(os.path.join(args.preview, f"{args.inst}_{note_name(midi)}.wav"), render_preview(y, loop))

    pack = {
        "version": PACK_VERSION, "key": args.inst, "name": cfg["name"], "sustain": cfg["sustain"],
        "sampleRate": OUT_SR, "referenceA4": 440.0, "channels": 1,
        "samples": samples,
    }
    with open(os.path.join(out_dir, "pack.json"), "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
        f.write("\n")
    total = sum(s["sizeBytes"] for s in samples)
    print(f"\n📦 instruments/{args.inst}/pack.json: {len(samples)}음 "
          f"({note_name(samples[0]['midi'])}~{note_name(samples[-1]['midi'])}), {total/1e6:.2f}MB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
