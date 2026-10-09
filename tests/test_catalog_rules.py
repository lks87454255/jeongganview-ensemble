# -*- coding: utf-8 -*-
"""python3 -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ensemble_common as ec  # noqa: E402
import build_ensemble as be  # noqa: E402


class CommonRulesTest(unittest.TestCase):
    def test_cycle_ms(self):
        self.assertEqual(ec.cycle_ms("18/8", 80), 13500)   # 진양조
        self.assertEqual(ec.cycle_ms("12/4", 60), 12000)   # 중모리
        self.assertEqual(ec.cycle_ms("12/8", 240), 3000)   # 굿거리
        self.assertEqual(ec.cycle_ms("4/4", 200), 1200)    # 휘모리
        self.assertIsNone(ec.cycle_ms("1/4", 40))          # 제례악: 장단 아님
        self.assertIsNone(ec.cycle_ms("60/8", 60))         # 60초 → 범위 밖
        self.assertIsNone(ec.cycle_ms(None, 100))
        self.assertIsNone(ec.cycle_ms("12/8", None))

    def test_mode(self):
        self.assertEqual(ec.mode_of("판소리"), "accompaniment")
        self.assertEqual(ec.mode_of("민요"), "accompaniment")
        self.assertEqual(ec.mode_of("산조"), "solo")
        self.assertEqual(ec.mode_of("풍류음악"), "ensemble")

    def test_jangdan(self):
        self.assertEqual(ec.jangdan_of("신쾌동류 거문고 산조 중 중중모리 4", "산조", "12/8", 140), ("중중모리", False))
        self.assertEqual(ec.jangdan_of("성금연류 가야금 산조 중 중모리 2", "산조", "12/4", 60), ("중모리", False))
        self.assertEqual(ec.jangdan_of("사랑가1", "판소리", "12/4", 80), ("중모리", True))
        self.assertEqual(ec.jangdan_of("이밤웬놈", "판소리", "12/8", 260), ("자진모리", True))
        self.assertEqual(ec.jangdan_of("x", "판소리", "12/8", 200), (None, False))  # 애매하면 추정 안 함
        self.assertEqual(ec.jangdan_of("도라지", "민요", "9/8", 100), ("세마치", True))
        self.assertEqual(ec.jangdan_of("상령산", "풍류음악", "60/8", 100), (None, False))
        self.assertEqual(ec.jangdan_of("까투리타령", "민요", "12/8", 200), (None, False))
        self.assertEqual(ec.jangdan_of("방아타령 1", "판소리", "12/8", 260), ("자진모리", True))

    def test_work_movement(self):
        self.assertEqual(ec.work_movement("신쾌동류 거문고 산조 중 진양조 17", "산조", "거문고 산조"),
                         ("신쾌동류 거문고 산조", "진양조 17"))
        self.assertEqual(ec.work_movement("현악영산회상(중광지곡) 상령산 2장", "풍류음악", "영산회상, 도드리"),
                         ("현악영산회상(중광지곡) 상령산", "2장"))
        self.assertEqual(ec.work_movement("평조회상(유초신지곡) 중 가락덜이 1장~3장", "풍류음악", ""),
                         ("평조회상(유초신지곡)", "가락덜이 1장~3장"))
        self.assertEqual(ec.work_movement("기생점고 2", "판소리", "춘향가"), ("춘향가", "기생점고 2"))
        self.assertEqual(ec.work_movement("한강수타령", "민요", "경기민요"), ("경기민요", "한강수타령"))
        self.assertEqual(ec.work_movement("비단길 2악장", "창작국악", "창작국악"), ("비단길", "2악장"))
        self.assertEqual(ec.work_movement("나를 따라 강동 갑세(1)", "창작국악", ""), ("나를 따라 강동 갑세", "(1)"))
        self.assertEqual(ec.work_movement("본령", "궁중음악", ""), ("본령", None))
        self.assertEqual(ec.work_movement("두레 - 2악장", "창작국악", ""), ("두레", "2악장"))

    def test_instruments(self):
        self.assertEqual(ec.split_instrument("대금2"), ("대금", 2))
        self.assertEqual(ec.split_instrument("피리1"), ("피리", 1))
        self.assertEqual(ec.instrument_info("소리북"), ("소리북", "soribuk", "타악"))
        self.assertEqual(ec.instrument_info("아쟁3")[:2], ("아쟁", "ajaeng"))
        self.assertEqual(ec.instrument_info("알수없음")[1:], (None, "기타"))

    def test_csv_helpers(self):
        self.assertEqual(ec.parse_moods("의례적인(2) | 장엄한(3)"),
                         [{"mood": "의례적인", "count": 2}, {"mood": "장엄한", "count": 3}])
        self.assertEqual(ec.parse_date("2025. 8. 5"), "2025-08-05")
        lab = ec.label_from_csv({"project_name": "0001_정악_풍류음악", "title": "t", "genreMajor": "정악",
                                 "genreSub": "풍류음악", "genreDetail": "d", "mood_count": "a(1)",
                                 "timeSignature": "60/8", "tempoBpm": "100", "western_key": "Ab / Fm",
                                 "recordDate": "2025. 8. 5"}, ["대금", "장구"])
        self.assertEqual(lab["master"]["tags"]["tempoBpm"], 100)
        self.assertEqual([s["file_name"] for s in lab["stems"]],
                         ["0001_정악_풍류음악_대금.wav", "0001_정악_풍류음악_장구.wav"])
        self.assertEqual(lab["stems"][1]["tags"]["instrumentMajor"], "타악기")

    def test_natural_key(self):
        self.assertEqual(sorted(["중모리 10", "중모리 2", "중모리 1"], key=ec.natural_key),
                         ["중모리 1", "중모리 2", "중모리 10"])


class NnlsTest(unittest.TestCase):
    def test_recovers_gains_and_clamps_negative(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((5000, 3))
        y = A @ np.array([0.5, 1.2, 0.0]) - 0.3 * A[:, 2]   # 셋째는 음수 → 0 으로
        g = be._nnls_from_gram(A.T @ A, A.T @ y)
        from scipy.optimize import nnls
        ref, _ = nnls(A, y)
        np.testing.assert_allclose(g, ref, atol=1e-8)
        self.assertEqual(g[2], 0.0)
        np.testing.assert_allclose(g[:2], [0.5, 1.2], atol=0.03)


class AlignTest(unittest.TestCase):
    def test_lag_estimate_and_delay(self):
        rng = np.random.default_rng(1)
        s = rng.standard_normal(6000)
        y = np.concatenate([np.zeros(8), s[:-8]])           # 마스터가 8표본 늦음
        self.assertAlmostEqual(be._estimate_lag(y, s, 300), 8.0, delta=0.05)
        self.assertAlmostEqual(be._estimate_lag(s, y, 300), -8.0, delta=0.05)
        np.testing.assert_allclose(be._delay(y, 8.0)[20:-20], s[20:-20], atol=1e-9)

    def test_analyze_mix_recovers_gains_with_offset(self):
        import tempfile
        import wave
        rng = np.random.default_rng(2)
        sr, n, lag = 48000, 48000 * 4, 64                   # 실제 원본과 같은 64표본(1.33ms) 지연
        a, b = rng.standard_normal(n) * 0.1, rng.standard_normal(n) * 0.1
        mix = np.concatenate([np.zeros(lag), (0.5 * a + 1.0 * b)[:-lag]])

        def write(path, x):
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())

        with tempfile.TemporaryDirectory() as d:
            paths = [os.path.join(d, f) for f in ("m.wav", "a.wav", "b.wav")]
            for p, x in zip(paths, (mix, a, b)):
                write(p, x)
            res = be.analyze_mix(be.Wav(paths[0]), [be.Wav(paths[1]), be.Wav(paths[2])])
        self.assertTrue(res["fitted"])
        self.assertAlmostEqual(res["lagMs"], 1.33, delta=0.05)
        np.testing.assert_allclose(res["gains"], [0.5, 1.0], atol=0.03)
        self.assertLess(res["mixResidualDb"], -20)


if __name__ == "__main__":
    unittest.main()
