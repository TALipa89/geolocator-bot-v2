"""Unit checks for the v2 build: verdict parsing, payload shapes, routing.

All inputs are synthetic. Run: python3 -m unittest test_main
"""
import base64
import io
import json
import unittest
import urllib.error

import main as geo


class VerdictParsing(unittest.TestCase):
    def test_plain_verdict(self):
        line = 'Sign in Cyrillic, cars on the right.\n{"lat": 55.75, "lon": 37.61, "confidence": 80}'
        self.assertEqual(geo.read_verdict(line), (55.75, 37.61, 80.0))

    def test_final_object_wins(self):
        text = '{"lat": 0, "lon": 0, "confidence": 5} actually {"lat": 1, "lon": 2, "confidence": 60}'
        self.assertEqual(geo.read_verdict(text), (1.0, 2.0, 60.0))

    def test_transposed_pair_fixed(self):
        self.assertEqual(geo.read_verdict('{"lat": 139.7, "lon": 35.7}'), (35.7, 139.7, 0.0))

    def test_fraction_becomes_percent(self):
        self.assertEqual(geo.read_verdict('{"lat": -33.9, "lon": 18.4, "confidence": 0.7}'), (-33.9, 18.4, 70.0))

    def test_long_key_names(self):
        self.assertEqual(geo.read_verdict('{"latitude": 48.85, "longitude": 2.35}'), (48.85, 2.35, 0.0))

    def test_loose_number_pair(self):
        self.assertEqual(geo.read_verdict("looks like 48.85, 2.35 to me"), (48.85, 2.35, 0.0))

    def test_no_verdict(self):
        self.assertIsNone(geo.read_verdict("no clue at all"))
        self.assertIsNone(geo.read_verdict('{"lat": 999, "lon": 999}'))
        self.assertIsNone(geo.read_verdict(""))


class CodeFallback(unittest.TestCase):
    def test_junk_bytes_use_fallback_centroid(self):
        self.assertEqual(geo.locate(b"definitely not a jpeg"), geo.CENTROIDS["fallback"])


class PayloadShape(unittest.TestCase):
    def test_anthropic_payload(self):
        doc = json.loads(geo.make_payload("anthropic", "claude-opus-5-5", b"\xff\xd8pic", "where?"))
        self.assertEqual(doc["model"], "claude-opus-5-5")
        self.assertEqual(doc["max_tokens"], geo.TOKEN_CAP)
        head = doc["messages"][0]["content"][0]
        self.assertEqual(head["source"]["media_type"], "image/jpeg")
        self.assertEqual(base64.b64decode(head["source"]["data"]), b"\xff\xd8pic")

    def test_openai_payload(self):
        doc = json.loads(geo.make_payload("openai", "gpt-6.1-sol", b"\xff\xd8pic", "where?"))
        self.assertEqual(doc["max_completion_tokens"], geo.TOKEN_CAP)
        link = doc["messages"][1]["content"][0]["image_url"]["url"]
        self.assertTrue(link.startswith("data:image/jpeg;base64,"))


class StubReply:
    def __init__(self, status, doc):
        self.status = status
        self.raw = json.dumps(doc).encode()

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class StubOpener:
    def __init__(self, fn):
        self.fn = fn
        self.seen = []

    def open(self, req, timeout=None):
        self.seen.append(req.full_url)
        return self.fn(req.full_url, len(self.seen))


def err(url, code, doc, headers=None):
    return urllib.error.HTTPError(url, code, "boom", headers or {}, io.BytesIO(json.dumps(doc).encode()))


class RouteBehaviour(unittest.TestCase):
    def setUp(self):
        self.keep = geo.OPENER

    def tearDown(self):
        geo.OPENER = self.keep

    def test_unauthorised_steps_down_route(self):
        def fn(url, n):
            if "messages" in url:
                raise err(url, 401, {"error": {"type": "authentication_error"}})
            return StubReply(200, {"choices": [{"message": {"content": '{"lat": 3, "lon": 4, "confidence": 90}'}}]})

        geo.OPENER = StubOpener(fn)
        state, text = geo.Gateway().ask(b"\xff\xd8", "where?")
        self.assertEqual(state, "ok")
        self.assertEqual(geo.read_verdict(text), (3.0, 4.0, 90.0))
        self.assertEqual(len(geo.OPENER.seen), 4)

    def test_budget_halts_everything(self):
        def fn(url, n):
            raise err(url, 429, {"error": {"type": "budget_exhausted"}}, {"x-should-retry": "false"})

        geo.OPENER = StubOpener(fn)
        gate = geo.Gateway()
        self.assertEqual(gate.ask(b"\xff\xd8", "where?")[0], "budget")
        self.assertEqual(gate.ask(b"\xff\xd8", "where?")[0], "budget")
        self.assertEqual(len(geo.OPENER.seen), 1)


if __name__ == "__main__":
    unittest.main()
