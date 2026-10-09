# jeongganview-ensemble

정간뷰 앱 "소리 마당"(악기별 합주)과 녹음 음색의 음원 에셋. 원본은 AI-Hub 71955 국악 합주곡(WAV, zip)이고 git 에는 변환 결과만 둔다.

## 파일 구조

```
ensemble-index.json          앱이 처음 받는 목록 (version 2: genres[] + v1 호환 categories[])
index/v2/{genre}.json        장르별 곡 요약 (pansori, minyo, sanjo, pungryu, daepungryu, gungjung, changjak)
ensemble/{곡}/
  {곡}.meta.json             곡 정보 + 파일 목록 (v2)
  {곡}.m4a                   마스터 (AAC 128k, 48kHz)
  {곡}.json                  라벨 JSON 원본 (있는 곡만)
  stems/{곡}_{악기}.m4a      악기별 음원
instruments/{key}/           녹음 음색 팩 (build_instruments.py)
curation.txt                 빌드할 곡 목록
```

## 명령

필요: Python 3.12 + numpy/scipy. ffmpeg 이 없으면 macOS `afconvert` 를 쓴다.

```bash
META="…/메타데이터.csv"            # 1.데이터/Other/Other.zip 안
ZIPS="…/2.데이터(NIA)"             # AI-Hub 다운로드 원형(zip) 루트

# 1) 곡 선정 (조각·6분 초과 제외, 장르별로 고르게)
python3 select_curation.py --meta-csv "$META" --zip-root "$ZIPS" > curation.txt

# 2) 빌드 (곡마다 zip 에서 임시로 풀고 끝나면 지움)
python3 build_ensemble.py --zip-root "$ZIPS" --meta-csv "$META" --select curation.txt
#   --song <project_name>  한 곡만     --reanalyze  기본 믹스만 다시 계산
#   --force                재인코딩    --tmp <dir>  임시 해제 위치

# 3) 인덱스 (내용이 같으면 파일을 다시 쓰지 않음)
python3 generate_index.py

# 4) 테스트
python3 -m unittest discover -s tests
```

## meta.json v2 주요 필드

| 필드 | 뜻 |
|---|---|
| `version` | 2 |
| `work` / `movement` | 작품 / 악장·대목 (제목에서 계산) |
| `genreMajor` / `genreSub` / `genreDetail` | 장르 (예: 민속악 / 판소리 / 춘향가) |
| `tempoBpm`, `timeSignature` | 원본 메타 값 |
| `jangdan`, `jangdanInferred` | 장단 이름, 제목에 없어 박자로 추정했으면 true |
| `cycleMs` | 장단 한 바퀴 = 박자 분자 × 60 / tempoBpm (1~40초 밖이면 null) |
| `mode` | `accompaniment`(판소리·민요) / `solo`(산조) / `ensemble`(그 외) |
| `leadKey` | solo 곡의 독주 스템 key |
| `fragment` | 60초 미만 조각 |
| `sampleRate` / `sourceSampleRate` | 출력 48000 / 원본 (96k 곡은 리샘플) |
| `quality` | `fitted`, `unitResidualDb`(단순 합 잔차), `mixResidualDb`(기본 믹스 잔차), `maxStemCorr`(스템 간 최대 상관), `silentStems` |
| `master`, `stems[]` | `fileName`, `url`, `sizeBytes`, `sha1` |
| `stems[].defaultGain` | 마스터에 맞춘 기본 볼륨 (0.05~1.0, 시간차 정렬 후 최소제곱) |
| `stems[].instrumentMajor` | 같은 악기 묶음 (대금2 → 대금) |

## 환경변수

- `ENSEMBLE_BASE_URL`: meta·인덱스에 적는 URL 접두사. 기본값은 이 저장소의 GitHub raw 주소. CDN 으로 옮길 때 이 값만 바꿔 빌드·인덱스를 다시 만든다.
