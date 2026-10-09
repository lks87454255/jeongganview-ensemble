#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ensemble.py (v2)
======================
AI-Hub 국악 합주곡(71955) 원본을 소리 마당(Stem Mixer) 에셋으로 변환합니다.

출력:
  ensemble/{project_name}/
  ├── {project_name}.json          ← 라벨 원본 그대로 복사 (라벨이 없는 곡은 메타데이터.csv 로 같은 형식 생성)
  ├── {project_name}.meta.json     ← 앱용 정보 (v2: 작품/장단/모드/기본 믹스 등)
  ├── {project_name}.m4a           ← master
  └── stems/{stem}.m4a

원본 두 가지 형식을 지원합니다.
  --source DIR   : 01.원천데이터/{곡}/*.wav + 02.라벨링데이터/*.json 구조 (풀린 폴더)
  --zip-root DIR : AI-Hub 다운로드 그대로 (…/2.데이터(NIA)/{Training,Validation}/01.원천데이터/*.zip)
                   곡 하나씩 임시 폴더에 풀고 끝나면 지운다.

v2 처리:
  - 96kHz 원본 → 48kHz 로 변환 (앱 믹서는 48kHz 하나로 동작)
  - 기본 믹스: 스템별 게인을 마스터에 최소제곱(음수 없음)으로 맞춰 stems[].defaultGain 에 저장
               (가장 큰 값 = 1.0 으로 정규화. 맞춤이 나쁘면 모두 1.0)
  - 품질 기록: 마스터와의 잔차(dB), 스템 간 최대 상관(새는 소리 지표)
  - 카탈로그 파생값: work / movement / jangdan / cycleMs / mode / fragment / leadKey

사용 예:
  python3 build_ensemble.py --zip-root "~/aihub/71955/.../2.데이터(NIA)" \
      --meta-csv /tmp/메타데이터.csv --select curation.txt
  python3 build_ensemble.py --source ~/Downloads/sample --song 0010_정악_풍류음악
"""

import argparse
import contextlib
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone

import numpy as np

from ensemble_common import (
    ENSEMBLE_DIR, FRAGMENT_MAX_MS, STEMS_DIR, TARGET_SAMPLE_RATE,
    asset_url, cycle_ms, instrument_info, jangdan_of, label_from_csv,
    load_meta_csv, mode_of, nfc, work_movement,
)

META_VERSION = 2
DURATION_TOLERANCE_MS = 10           # 스템 간 길이 차이 허용
ANALYSIS_RATE = 6000                 # 기본 믹스 계산용 저역 표본화 (게인은 대역 무관 상수)
MIN_GAIN = 0.05                      # 정규화 후 최저 기본 볼륨
FIT_IMPROVE_DB = 1.0                 # 단순 합보다 이만큼 이상 나아질 때만 계산 게인 사용
MAX_ALIGN_MS = 50                    # 마스터 ↔ 스템 합 시간차 탐색 범위


def log(msg: str) -> None:
    print(msg, flush=True)


class BuildError(Exception):
    pass


# ── WAV 헤더/샘플 읽기 (24bit, 확장 청크 포함) ─────────────────
class Wav:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                raise BuildError(f"WAV 형식이 아닙니다: {path}")
            fmt = None
            while True:
                h = f.read(8)
                if len(h) < 8:
                    break
                cid, size = h[:4], struct.unpack("<I", h[4:])[0]
                if cid == b"fmt ":
                    body = f.read(size)
                    tag, ch, sr = struct.unpack("<HHI", body[:8])
                    bits = struct.unpack("<H", body[14:16])[0]
                    if tag == 0xFFFE and len(body) >= 26:
                        tag = struct.unpack("<H", body[24:26])[0]
                    fmt = (tag, ch, sr, bits)
                    if size & 1:
                        f.read(1)
                elif cid == b"data":
                    self.data_offset = f.tell()
                    self.data_size = size
                    break
                else:
                    f.seek(size + (size & 1), 1)
        if fmt is None or not hasattr(self, "data_offset"):
            raise BuildError(f"WAV 청크를 찾을 수 없습니다: {path}")
        self.tag, self.channels, self.sample_rate, self.bits = fmt
        if self.tag not in (1, 3) or self.bits not in (16, 24, 32):
            raise BuildError(f"지원하지 않는 WAV 형식(tag={self.tag}, bits={self.bits}): {path}")
        real = os.path.getsize(path) - self.data_offset
        self.data_size = min(self.data_size, real)  # 크기 필드가 잘못된 파일 대비
        self.frame_bytes = self.channels * self.bits // 8
        self.frames = self.data_size // self.frame_bytes

    @property
    def duration_ms(self) -> int:
        return round(self.frames * 1000 / self.sample_rate)

    def mono_blocks(self, block_frames: int):
        """(frames,) float32 모노 블록을 순서대로 반환."""
        with open(self.path, "rb") as f:
            f.seek(self.data_offset)
            pos = 0
            while pos < self.frames:
                n = min(block_frames, self.frames - pos)
                raw = np.frombuffer(f.read(n * self.frame_bytes), dtype=np.uint8)
                n = len(raw) // self.frame_bytes
                if n == 0:
                    break
                yield self._to_float(raw[:n * self.frame_bytes]).reshape(n, self.channels).mean(axis=1)
                pos += n

    def _to_float(self, raw: np.ndarray) -> np.ndarray:
        if self.tag == 3:
            return raw.view(np.float32 if self.bits == 32 else np.float64).astype(np.float32)
        if self.bits == 16:
            return raw.view("<i2").astype(np.float32) / 32768.0
        if self.bits == 32:
            return raw.view("<i4").astype(np.float32) / 2147483648.0
        b = raw.reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = (v ^ 0x800000) - 0x800000
        return v.astype(np.float32) / 8388608.0


# ── 기본 믹스(마스터 균형) 계산 ────────────────────────────────
def _nnls_from_gram(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min ||A g - y||², g ≥ 0 를 정규방정식(G=AᵀA, b=Aᵀy)만으로 푼다 (활성집합 Lawson-Hanson)."""
    k = len(b)
    g = np.zeros(k)
    passive = np.zeros(k, dtype=bool)
    for _ in range(3 * k + 10):
        w = b - G @ g
        cand = np.where(~passive & (w > 1e-12))[0]
        if len(cand) == 0:
            break
        passive[cand[np.argmax(w[cand])]] = True
        while True:
            idx = np.where(passive)[0]
            s = np.zeros(k)
            s[idx] = np.linalg.solve(G[np.ix_(idx, idx)], b[idx])
            if np.all(s[idx] > 0):
                g = s
                break
            neg = idx[s[idx] <= 0]
            alpha = np.min(g[neg] / (g[neg] - s[neg] + 1e-30))
            g = g + alpha * (s - g)
            passive &= g > 1e-12
    return g


def _decimated(w: Wav, dec: int) -> np.ndarray:
    """모노 → dec 표본씩 평균한 저역 신호 (원본 블록은 dec 의 배수라 경계가 맞는다)."""
    out = []
    for x in w.mono_blocks(dec * 65536):
        n = (len(x) // dec) * dec
        if n:
            out.append(x[:n].reshape(-1, dec).mean(axis=1))
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def _estimate_lag(y: np.ndarray, s: np.ndarray, max_lag: int) -> float:
    """y(t) ≈ s(t - lag) 인 lag (표본 단위, 포물선 보간으로 소수까지, |lag| ≤ max_lag)."""
    n = min(len(y), len(s))
    if n < 4 * max_lag + 16 or max_lag <= 0:
        return 0.0
    size = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(y[:n], size) * np.conj(np.fft.rfft(s[:n], size)), size)
    cand = np.concatenate([c[-max_lag:], c[:max_lag + 1]])       # lag -max..+max
    i = int(np.argmax(cand))
    lag = float(i - max_lag)
    if 0 < i < len(cand) - 1:
        a, m, z = cand[i - 1], cand[i], cand[i + 1]
        den = a - 2 * m + z
        if den < 0:
            lag += 0.5 * (a - z) / den
    return lag


def _delay(x: np.ndarray, d: float) -> np.ndarray:
    """x 를 d 표본만큼 앞당긴다 (out[t] = x[t + d], 소수 지연은 FFT 위상)."""
    n = len(x)
    size = 1 << int(np.ceil(np.log2(n + 2 * (abs(int(d)) + 2))))
    f = np.fft.rfftfreq(size)
    return np.fft.irfft(np.fft.rfft(x, size) * np.exp(2j * np.pi * f * d), size)[:n]


def analyze_mix(master: Wav, stems: list) -> dict:
    """스템별 기본 게인 + 품질 지표.

    저역 표본화한 신호로 계산(긴 곡도 메모리 작음). 일부 원본은 마스터가 스템 합보다
    수 ms 어긋나 있어(0401 대풍류, 0649 적벽가 등 ≈1.4ms) 그대로 맞추면 잔차가 0dB 근처가
    되므로, 교차상관으로 시간차를 찾아 마스터를 정렬한 뒤 맞춘다.
    """
    k = len(stems)
    dec = max(1, master.sample_rate // ANALYSIS_RATE)
    rate = master.sample_rate / dec
    yv = _decimated(master, dec)
    cols = []
    for s in stems:
        x = _decimated(s, dec)
        x = x[:len(yv)] if len(x) >= len(yv) else np.concatenate([x, np.zeros(len(yv) - len(x))])
        cols.append(x)
    A = np.stack(cols, axis=1) if k else np.zeros((len(yv), 0))

    lag = _estimate_lag(yv, A.sum(axis=1), int(rate * MAX_ALIGN_MS / 1000)) if k else 0.0
    res = _fit_gains(yv, A)
    if abs(lag) >= 0.05:   # 마스터를 lag 만큼 당겨 정렬(순환 경계 제거), 실제로 나아질 때만 채택
        e = int(np.ceil(abs(lag))) + 1
        aligned = _fit_gains(_delay(yv, lag)[e:len(yv) - e], A[e:len(A) - e])
        if aligned["mixResidualDb"] < res["mixResidualDb"] - FIT_IMPROVE_DB:
            res = aligned
        else:
            lag = 0.0
    else:
        lag = 0.0
    res["lagMs"] = round(lag * 1000 / rate, 2)
    return res


def _fit_gains(yv: np.ndarray, A: np.ndarray) -> dict:
    k = A.shape[1]
    G = A.T @ A
    b = A.T @ yv
    yy = float(yv @ yv)

    energy = np.diag(G).copy()
    alive = energy > 1e-9 * max(1e-30, energy.max())
    ridge = 1e-9 * np.trace(G) / max(1, k)
    g = np.ones(k)
    if alive.any():
        Gi = G[np.ix_(alive, alive)] + ridge * np.eye(int(alive.sum()))
        g_alive = _nnls_from_gram(Gi, b[alive])
        g[alive] = g_alive

    def residual_db(gv):
        r = yy - 2 * gv @ b + gv @ G @ gv
        return 10 * np.log10(max(r, 1e-20) / max(yy, 1e-20))

    unit_db = residual_db(np.ones(k))
    fit_db = residual_db(g)
    use_fit = alive.any() and fit_db < unit_db - FIT_IMPROVE_DB and g[alive].max() > 0
    if use_fit:
        gains = g / g[alive].max()
        gains = np.where(alive, np.clip(gains, MIN_GAIN, 1.0), 1.0)
    else:
        gains = np.ones(k)

    d = np.sqrt(np.maximum(energy, 1e-30))
    corr = G / np.outer(d, d)
    np.fill_diagonal(corr, 0.0)
    max_corr = float(np.abs(corr[np.ix_(alive, alive)]).max()) if alive.sum() > 1 else 0.0
    return {
        "gains": [round(float(x), 3) for x in gains],
        "fitted": bool(use_fit),
        "unitResidualDb": round(float(unit_db), 1),
        "mixResidualDb": round(float(fit_db if use_fit else unit_db), 1),
        "maxStemCorr": round(max_corr, 3),
        "silentStems": int((~alive).sum()),
    }


# ── 인코딩 ────────────────────────────────────────────────────
def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def encode_aac(src: str, dst: str, bitrate_k: int, src_rate: int) -> None:
    """WAV → AAC(m4a), 출력 48kHz. 임시 파일에 쓴 뒤 rename (중단 시 깨진 파일 방지)."""
    tmp = dst + ".tmp.m4a"
    if os.path.exists(tmp):
        os.remove(tmp)
    if which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ar", str(TARGET_SAMPLE_RATE),
               "-c:a", "aac", "-b:a", f"{bitrate_k}k", "-movflags", "+faststart", tmp]
    elif which("afconvert"):
        fmt = "aac" if src_rate == TARGET_SAMPLE_RATE else f"aac@{TARGET_SAMPLE_RATE}"
        cmd = ["afconvert", "-f", "m4af", "-d", fmt, "-b", str(bitrate_k * 1000), src, tmp]
    else:
        raise BuildError("ffmpeg 또는 afconvert 가 필요합니다.")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise BuildError(f"인코딩 실패: {src}\n{e.stderr}")
    os.replace(tmp, dst)


def m4a_duration_ms(path: str):
    if which("afinfo"):
        out = subprocess.run(["afinfo", path], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "estimated duration" in line:
                return round(float(line.split(":")[1].split()[0]) * 1000)
    return None


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_if_changed(path: str, data: dict) -> bool:
    """generated 를 제외한 내용이 같으면 쓰지 않는다(sha1/ETag/git diff 안정화)."""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            old.pop("generated", None)
            if old == {k: v for k, v in data.items() if k != "generated"}:
                return False
        except (OSError, ValueError):
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return True


def needs_encode(src: str, dst: str, force: bool) -> bool:
    if force or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        return True
    return os.path.getmtime(src) > os.path.getmtime(dst)


# ── 원본 공급자 ───────────────────────────────────────────────
def _index_files(root: str) -> dict:
    out = {}
    for dirpath, _d, files in os.walk(root):
        for f in files:
            if not f.startswith("."):
                out.setdefault(nfc(f), os.path.join(dirpath, f))
    return out


def _wav_stem_suffix(project: str, file_name: str):
    """'0001_정악_풍류음악_대금.wav' → '대금', master('0001_정악_풍류음악.wav') → None"""
    base = nfc(os.path.splitext(os.path.basename(file_name))[0])
    if base == project:
        return None
    return base[len(project) + 1:] if base.startswith(project + "_") else base


class DirSource:
    """01.원천데이터/{곡}/*.wav + 02.라벨링데이터/**/*.json"""

    def __init__(self, root: str):
        root = os.path.expanduser(root)
        sub = {nfc(d): os.path.join(root, d) for d in os.listdir(root)}
        self.raw_root = sub.get("01.원천데이터")
        label_root = sub.get("02.라벨링데이터")
        if not self.raw_root:
            raise BuildError(f"{root} 아래에 01.원천데이터 폴더가 필요합니다.")
        self.labels = {}
        if label_root:
            for p in glob.glob(os.path.join(label_root, "**", "*.json"), recursive=True):
                self.labels[nfc(os.path.splitext(os.path.basename(p))[0])] = p
        self.dirs = {}
        for dirpath, dirs, _f in os.walk(self.raw_root):
            for d in dirs:
                self.dirs.setdefault(nfc(d), os.path.join(dirpath, d))

    def songs(self):
        return sorted(self.dirs)

    def label_bytes(self, song):
        p = self.labels.get(song)
        return open(p, "rb").read() if p else None

    @contextlib.contextmanager
    def open_song(self, song):
        d = self.dirs.get(song)
        if not d:
            raise BuildError(f"원천데이터 폴더를 찾을 수 없습니다: {song}")
        yield _index_files(d)


class ZipSource:
    """AI-Hub 다운로드 원형: */01.원천데이터/{TS|VS}_{곡}.zip, */02.라벨링데이터/{TL|VL}_{곡}.zip"""

    def __init__(self, root: str, tmp_root: str = None):
        root = os.path.expanduser(root)
        self.tmp_root = tmp_root
        self.zips, self.label_zips = {}, {}
        for p in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
            name = nfc(os.path.basename(p))[:-4]
            parent = nfc(os.path.basename(os.path.dirname(p)))
            if len(name) < 4 or name[2] != "_":
                continue
            project = name[3:]
            if parent == "01.원천데이터":
                self.zips[project] = p
            elif parent == "02.라벨링데이터":
                self.label_zips[project] = p
        if not self.zips:
            raise BuildError(f"원천 zip 을 찾을 수 없습니다: {root}")

    def songs(self):
        return sorted(self.zips)

    @staticmethod
    def _name(info: zipfile.ZipInfo) -> str:
        if info.flag_bits & 0x800:
            return nfc(info.filename)
        try:
            return nfc(info.filename.encode("cp437").decode("utf-8"))
        except UnicodeError:
            return nfc(info.filename.encode("cp437").decode("cp949", "ignore"))

    def label_bytes(self, song):
        p = self.label_zips.get(song)
        if not p:
            return None
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if self._name(i).lower().endswith(".json"):
                    return z.read(i)
        return None

    def wav_names(self, song):
        with zipfile.ZipFile(self.zips[song]) as z:
            return [os.path.basename(self._name(i)) for i in z.infolist()
                    if self._name(i).lower().endswith(".wav")]

    @contextlib.contextmanager
    def open_song(self, song):
        p = self.zips.get(song)
        if not p:
            raise BuildError(f"원천 zip 을 찾을 수 없습니다: {song}")
        tmp = tempfile.mkdtemp(prefix="ensemble_", dir=self.tmp_root)
        try:
            files = {}
            with zipfile.ZipFile(p) as z:
                for i in z.infolist():
                    name = os.path.basename(self._name(i))
                    if not name.lower().endswith(".wav"):
                        continue
                    dst = os.path.join(tmp, name)
                    with z.open(i) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
                    # 원본 시각 유지 → 다음 빌드에서 needs_encode 가 불필요한 재인코딩을 하지 않음
                    ts = time.mktime(i.date_time + (0, 0, -1))
                    os.utime(dst, (ts, ts))
                    files[name] = dst
            yield files
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── 곡 1개 처리 ────────────────────────────────────────────────
def resolve_label(song: str, source, csv_rows: dict, wav_names: list):
    """(label dict, label 원본 bytes or None)"""
    raw = source.label_bytes(song)
    if raw:
        label = json.loads(raw.decode("utf-8-sig"))
        if nfc(label.get("project_name", "")) != song:
            raise BuildError(f"라벨의 project_name({label.get('project_name')}) 과 곡({song}) 이 다릅니다.")
        return label, raw
    row = csv_rows.get(song)
    if not row:
        raise BuildError(f"라벨 JSON 도 메타데이터.csv 행도 없습니다: {song} (--meta-csv 지정 필요)")
    stems = sorted(s for s in (_wav_stem_suffix(song, n) for n in wav_names) if s)
    return label_from_csv(row, stems), None


def stem_key(name: str, idx: int, used: set) -> tuple:
    base, key, group = instrument_info(name)
    if key is None:
        key = f"stem{idx + 1:02d}"
        log(f"  ⚠️  매핑 없는 악기: '{name}' → key={key}, group=기타 (ensemble_common.INSTRUMENTS 에 추가 권장)")
    k, n = key, 2
    while k in used:              # 대금, 대금2 → daegeum, daegeum_2
        k = f"{key}_{n}"
        n += 1
    used.add(k)
    return k, base, group


def build_song(song: str, source, csv_rows: dict, out_root: str, bitrate_k: int,
               force: bool, analyze: bool) -> dict:
    log(f"\n🎼 {song}")
    with source.open_song(song) as files:
        label, label_raw = resolve_label(song, source, csv_rows, list(files))
        master = label.get("master") or {}
        stems = label.get("stems") or []
        if not master.get("file_name") or not stems:
            raise BuildError("라벨에 master/stems 정보가 없습니다.")

        def src_of(file_name: str) -> str:
            p = files.get(nfc(os.path.basename(file_name)))
            if not p:
                raise BuildError(f"원천 음원을 찾을 수 없습니다: {file_name}")
            return p

        # 1) 소스 검증
        master_src = src_of(master["file_name"])
        mw = Wav(master_src)
        stem_items = []
        for st in stems:
            p = src_of(st["file_name"])
            w = Wav(p)
            if w.sample_rate != mw.sample_rate:
                raise BuildError(f"샘플레이트 불일치: {st['file_name']} {w.sample_rate} != {mw.sample_rate}")
            if abs(w.duration_ms - mw.duration_ms) > DURATION_TOLERANCE_MS:
                raise BuildError(f"길이 불일치: {st['file_name']} {w.duration_ms}ms != {mw.duration_ms}ms")
            stem_items.append((st, p, w))
        log(f"  ✅ 소스: stem {len(stems)}개, {mw.sample_rate}Hz, {mw.channels}ch, {mw.bits}bit, "
            f"{mw.duration_ms / 1000:.1f}s" + ("  (→ 48kHz 변환)" if mw.sample_rate != TARGET_SAMPLE_RATE else ""))

        # 2) 기본 믹스 분석
        song_dir = os.path.join(out_root, ENSEMBLE_DIR, song)
        meta_path = os.path.join(song_dir, f"{song}.meta.json")
        mix = None
        if not analyze and os.path.isfile(meta_path):   # 이전 분석 재사용
            try:
                old = json.load(open(meta_path, encoding="utf-8"))
                if old.get("quality") and len(old.get("stems") or []) == len(stems):
                    mix = {**old["quality"], "gains": [s.get("defaultGain", 1.0) for s in old["stems"]]}
            except (OSError, ValueError):
                mix = None
        if mix is None:
            t0 = time.time()
            mix = analyze_mix(mw, [w for _s, _p, w in stem_items])
            log(f"  📊 기본 믹스: {'맞춤' if mix['fitted'] else '단순합'} "
                f"잔차 {mix['unitResidualDb']} → {mix['mixResidualDb']}dB, "
                f"스템 최대 상관 {mix['maxStemCorr']}, 마스터 시간차 {mix['lagMs']}ms "
                f"({time.time() - t0:.1f}s)")
            if mix["maxStemCorr"] > 0.3:
                log("  ⚠️  스템 간 상관이 큽니다 (다른 악기 소리가 섞였을 수 있음)")

        # 3) 출력 폴더 + 라벨
        os.makedirs(os.path.join(song_dir, STEMS_DIR), exist_ok=True)
        label_name = f"{song}.json"
        label_path = os.path.join(song_dir, label_name)
        if label_raw is not None:
            if not os.path.isfile(label_path) or open(label_path, "rb").read() != label_raw:
                with open(label_path, "wb") as f:
                    f.write(label_raw)     # 원본 바이트 그대로
        else:
            write_json_if_changed(label_path, label)

        # 4) 인코딩
        def encode(src: str, w: Wav, rel_parts: list) -> dict:
            dst = os.path.join(song_dir, *rel_parts)
            if needs_encode(src, dst, force):
                log(f"  🔄 인코딩: {'/'.join(rel_parts)}")
                encode_aac(src, dst, bitrate_k, w.sample_rate)
            return {"url": asset_url(ENSEMBLE_DIR, song, *rel_parts),
                    "sizeBytes": os.path.getsize(dst), "sha1": sha1_of(dst)}

        master_name = nfc(os.path.splitext(os.path.basename(master["file_name"]))[0]) + ".m4a"
        master_entry = {"fileName": master_name, **encode(master_src, mw, [master_name])}
        out_ms = m4a_duration_ms(os.path.join(song_dir, master_name))

        used = set()
        stem_entries = []
        for idx, (st, src, w) in enumerate(stem_items):
            tags = st.get("tags") or {}
            name = nfc(tags.get("instrumentSub") or tags.get("instrumentMajor") or f"악기{idx + 1}")
            key, base, group = stem_key(name, idx, used)
            file_name = nfc(os.path.splitext(os.path.basename(st["file_name"]))[0]) + ".m4a"
            stem_entries.append({
                "key": key,
                "name": name,
                "baseName": base,
                "group": group,
                "instrumentMajor": tags.get("instrumentMajor"),
                "sourceFileName": nfc(os.path.basename(st["file_name"])),
                "fileName": file_name,
                **encode(src, w, [STEMS_DIR, file_name]),
                "defaultGain": mix["gains"][idx],
                "pan": 0.0,
            })

    expected = {e["fileName"] for e in stem_entries}
    for f in os.listdir(os.path.join(song_dir, STEMS_DIR)):
        if nfc(f) not in expected and not f.startswith("."):
            log(f"  🗑️  라벨에 없는 stem 삭제: {f}")
            os.remove(os.path.join(song_dir, STEMS_DIR, f))

    # 5) meta.json
    mt = master.get("tags") or {}
    title, genre_sub, genre_detail = mt.get("title"), mt.get("genreSub"), mt.get("genreDetail")
    ts, bpm = mt.get("timeSignature"), mt.get("tempoBpm")
    work, movement = work_movement(title, genre_sub, genre_detail)
    jangdan, inferred = jangdan_of(title, genre_sub, ts, bpm)
    mode = mode_of(genre_sub)
    duration_ms = mw.duration_ms
    lead = None
    if mode == "solo":
        melodic = [s for s in stem_entries if s["group"] != "타악"]
        lead = melodic[0]["key"] if len(melodic) == 1 else None
    meta = {
        "version": META_VERSION,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "id": song,
        "title": title,
        "work": work,
        "movement": movement,
        "performers": mt.get("performers"),
        "genreMajor": mt.get("genreMajor"),
        "genreSub": genre_sub,
        "genreDetail": genre_detail,
        "tempoBpm": bpm,
        "timeSignature": ts,
        "jangdan": jangdan,
        "jangdanInferred": inferred,
        "cycleMs": cycle_ms(ts, bpm),
        "mode": mode,
        "leadKey": lead,
        "fragment": duration_ms < FRAGMENT_MAX_MS,
        "westernKey": mt.get("western_key"),
        "recordDate": mt.get("recordDate"),
        "moods": [m.get("mood") for m in (mt.get("moods") or []) if m.get("mood")],
        "sampleRate": TARGET_SAMPLE_RATE,
        "sourceSampleRate": mw.sample_rate,
        "channels": 2,
        "durationMs": duration_ms,
        "codec": "aac",
        "bitrateKbps": bitrate_k,
        "labelUrl": asset_url(ENSEMBLE_DIR, song, label_name),
        "quality": {k: mix[k] for k in ("fitted", "unitResidualDb", "mixResidualDb", "maxStemCorr", "silentStems")},
        "master": master_entry,
        "stems": stem_entries,
    }
    if out_ms is not None and abs(out_ms - duration_ms) > 100:
        log(f"  ⚠️  인코딩 결과 길이 차이: {out_ms}ms vs 원본 {duration_ms}ms")
    log("  ✏️  meta.json 갱신" if write_json_if_changed(meta_path, meta) else "  ⏭️  meta.json 변경 없음")
    total = master_entry["sizeBytes"] + sum(s["sizeBytes"] for s in stem_entries)
    log(f"  📝 {work} / {movement or '-'} · {jangdan or '-'} · {mode} · 합계 {total / 1_000_000:.1f}MB")
    return meta


def read_select(path: str) -> list:
    out = []
    for line in open(os.path.expanduser(path), encoding="utf-8"):
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(nfc(s.split()[0]))
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="AI-Hub 원본 → 소리 마당 에셋 변환 (v2)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="01.원천데이터 / 02.라벨링데이터 를 포함한 폴더")
    src.add_argument("--zip-root", help="AI-Hub 다운로드 원형(zip) 루트, 예: …/2.데이터(NIA)")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--song", action="append", help="project_name (여러 번 지정 가능)")
    sel.add_argument("--select", help="곡 목록 파일 (한 줄에 project_name, # 주석)")
    sel.add_argument("--all", action="store_true", help="원본의 모든 곡")
    ap.add_argument("--meta-csv", help="메타데이터.csv (라벨 JSON 이 없는 곡의 정보)")
    ap.add_argument("--out", default=here, help="레포 루트 (기본: 스크립트 위치)")
    ap.add_argument("--bitrate", type=int, default=128, help="AAC 비트레이트 kbps (기본 128)")
    ap.add_argument("--force", action="store_true", help="m4a 재인코딩")
    ap.add_argument("--reanalyze", action="store_true", help="기본 믹스 다시 계산 (기본: 이전 결과 재사용)")
    ap.add_argument("--tmp", help="zip 임시 해제 위치 (기본: 시스템 임시 폴더)")
    args = ap.parse_args()

    try:
        source = DirSource(args.source) if args.source else ZipSource(args.zip_root, args.tmp)
    except BuildError as e:
        raise SystemExit(f"❌ {e}")
    csv_rows = load_meta_csv(os.path.expanduser(args.meta_csv)) if args.meta_csv else {}
    songs = source.songs() if args.all else (read_select(args.select) if args.select else [nfc(s) for s in args.song])
    if not songs:
        raise SystemExit("❌ 처리할 곡이 없습니다.")

    log(f"인코더: {'ffmpeg' if which('ffmpeg') else 'afconvert'} / {args.bitrate}kbps / "
        f"곡 {len(songs)}개 / 출력: {args.out}")
    failed = []
    t0 = time.time()
    for i, song in enumerate(songs, 1):
        log(f"\n[{i}/{len(songs)}]")
        try:
            build_song(song, source, csv_rows, args.out, args.bitrate, args.force,
                       analyze=args.reanalyze or args.force)
        except (BuildError, OSError, ValueError) as e:
            log(f"  ❌ 실패: {e}")
            failed.append((song, str(e)))
    log(f"\n완료: 성공 {len(songs) - len(failed)} / 실패 {len(failed)}  ({(time.time() - t0) / 60:.1f}분)")
    for s, e in failed:
        log(f"  ❌ {s}: {e.splitlines()[0]}")
    log("다음 단계 → python3 generate_index.py")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
