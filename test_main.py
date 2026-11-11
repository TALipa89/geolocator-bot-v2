"""Focused checks for main.py: answer parsing, request shape, route state.

Synthetic inputs only (no photos); run with: python3 -m unittest test_main
"""
import base64
import io
import json
import unittest
import urllib.error

import main as bot


class ParseGuess(unittest.TestCase):
    def test_final_json_line(self):
        text = 'Cyrillic sign, right-hand traffic.\n{"lat": 55.75, "lon": 37.61, "confidence": 80}'
        self.assertEqual(bot.parse_guess(text), (55.75, 37.61, 80.0))

    def test_last_json_wins(self):
        text = '{"lat": 0, "lon": 0, "confidence": 5} no, rather {"lat": 1, "lon": 2, "confidence": 60}'
        self.assertEqual(bot.parse_guess(text), (1.0, 2.0, 60.0))

    def test_swapped_pair_unswapped(self):
        self.assertEqual(bot.parse_guess('{"lat": 139.7, "lon": 35.7}'), (35.7, 139.7, 0.0))

    def test_fraction_confidence(self):
        self.assertEqual(bot.parse_guess('{"lat": -33.9, "lon": 18.4, "confidence": 0.7}'),
                         (-33.9, 18.4, 70.0))

    def test_bare_numbers_last_pair(self):
        self.assertEqual(bot.parse_guess("somewhere near 48.85, 2.35 I think"), (48.85, 2.35, 0.0))

    def test_garbage_and_out_of_range(self):
        self.assertIsNone(bot.parse_guess("no idea"))
        self.assertIsNone(bot.parse_guess('{"lat": 999, "lon": 999}'))
        self.assertIsNone(bot.parse_guess(""))


class Fallback(unittest.TestCase):
    def test_non_jpeg_falls_back(self):
        self.assertEqual(bot.locate(b"not a jpeg"), bot.CENTROIDS["fallback"])


class RequestBody(unittest.TestCase):
    def test_anthropic_shape(self):
        body = json.loads(bot._request_body("anthropic", "claude-opus-5-5", b"\xff\xd8x", "q"))
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["max_tokens"], bot.MAX_TOKENS)
        img = body["messages"][0]["content"][0]
        self.assertEqual(img["source"]["type"], "base64")
        self.assertEqual(base64.b64decode(img["source"]["data"]), b"\xff\xd8x")

    def test_openai_shape(self):
        body = json.loads(bot._request_body("openai", "gpt-6.1-sol", b"\xff\xd8x", "q"))
        self.assertEqual(body["max_completion_tokens"], bot.MAX_TOKENS)
        url = body["messages"][1]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Dispatches on the request URL: handler(url, count) -> FakeResponse or raise."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def open(self, req, timeout=None):
        self.calls.append(req.full_url)
        return self.handler(req.full_url, len(self.calls))


def http_error(url, code, payload, headers=None):
    return urllib.error.HTTPError(url, code, "err", headers or {}, io.BytesIO(
        json.dumps(payload).encode()))


class RouteState(unittest.TestCase):
    def setUp(self):
        self.saved = bot.OPENER

    def tearDown(self):
        bot.OPENER = self.saved

    def test_401_advances_to_openai(self):
        def handler(url, n):
            if "messages" in url:
                raise http_error(url, 401, {"error": {"type": "authentication_error"}})
            return FakeResponse(200, {"choices": [{"message": {
                "content": '{"lat": 3, "lon": 4, "confidence": 90}'}}]})

        bot.OPENER = FakeOpener(handler)
        kind, text = bot.ModelRoute().ask(b"\xff\xd8", "q")
        self.assertEqual(kind, "ok")
        self.assertEqual(bot.parse_guess(text), (3.0, 4.0, 90.0))
        # 3 anthropic candidates refused, then the first OpenAI one answered.
        self.assertEqual(len(bot.OPENER.calls), 4)

    def test_budget_stops_all_calls(self):
        def handler(url, n):
            raise http_error(url, 429, {"error": {"type": "budget_exhausted"}},
                             {"x-should-retry": "false"})

        bot.OPENER = FakeOpener(handler)
        route = bot.ModelRoute()
        self.assertEqual(route.ask(b"\xff\xd8", "q")[0], "budget")
        self.assertEqual(route.ask(b"\xff\xd8", "q")[0], "budget")
        self.assertEqual(len(bot.OPENER.calls), 1)


if __name__ == "__main__":
    unittest.main()
