"""photo-geolocator-01 entry v2: vision-model geolocator, code fallback.

For each photo it asks an allowlisted vision model through the gateway
(Anthropic Messages first, OpenAI Chat Completions if that vendor has no key)
with a fixed GeoGuessr-style clue prompt, reads {"lat", "lon", "confidence"}
out of the answer, banks confident answers at once, then gives the least
confident photos one challenge second look. Any photo the model route cannot
place (no key, spent budget, refusal, unparseable answer) falls back to the
v1 code-only locator: JPEG DC colour-climate bands (decode only the DC
coefficients, average sky/ground rows in YCbCr, map snow/desert/tropics/
temperate colour to that band's land centroid; no trained weights, no photo
table). Every photo is answered exactly once. Deterministic: requests are a
pure function of the photo bytes, the fixed prompts and earlier answers, sent
in photo order, one at a time (a single call's worst case is then the only
reservation against the $1.00 trial budget).
Run: python3 -P -s /bot/main.py  (MAKEFUN_ENV_URL names the env)
"""
import base64
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_json(url, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def http_bytes(url):
    with OPENER.open(url, timeout=120) as res:
        return res.read()


# ---------------- JPEG DC extraction (baseline, interleaved) ----------------

class JpegError(Exception):
    pass


def build_huffman(counts, symbols):
    table = {}
    code = 0
    pos = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            table[(length, code)] = symbols[pos]
            pos += 1
            code += 1
        code <<= 1
    return table


class BitReader:
    def __init__(self, data, pos):
        self.data = data
        self.pos = pos
        self.buf = 0
        self.bits = 0

    def fill(self):
        data = self.data
        while self.bits <= 56 and self.pos < len(data):
            byte = data[self.pos]
            if byte == 0xFF:
                nxt = data[self.pos + 1] if self.pos + 1 < len(data) else 0
                if nxt == 0x00:
                    self.pos += 2  # stuffed 0xFF: the FF itself is data
                elif nxt == 0xFF:
                    self.pos += 1  # fill byte before a marker: skip it
                    continue
                else:
                    return False  # a real marker: entropy data is over
            else:
                self.pos += 1
            self.buf = (self.buf << 8) | byte
            self.bits += 8
        return self.bits > 0

    def get_bits(self, count):
        if count == 0:
            return 0
        if self.bits < count and not self.fill():
            raise JpegError("entropy data ended early")
        if self.bits < count:
            raise JpegError("entropy data ended early")
        shift = self.bits - count
        value = (self.buf >> shift) & ((1 << count) - 1)
        self.buf &= (1 << shift) - 1
        self.bits = shift
        return value

    def decode(self, table):
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.get_bits(1)
            hit = table.get((length, code))
            if hit is not None:
                return hit
        raise JpegError("bad huffman code")


def receive_extend(reader, count):
    if count == 0:
        return 0
    value = reader.get_bits(count)
    if value < (1 << (count - 1)):
        value -= (1 << count) - 1
    return value


def parse_jpeg_dc(data):
    """Return (y_rows, cb_avg, cr_avg): per-row mean Y DC, mean Cb/Cr DC."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise JpegError("not a jpeg")
    pos = 2
    quant = {}
    huff_dc = {}
    huff_ac = {}
    comps = {}
    width = height = 0
    restart_interval = 0
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise JpegError("bad marker")
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        size = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        seg = data[pos + 4:pos + 2 + size]
        if marker == 0xDB:
            off = 0
            while off < len(seg):
                pq, tq = seg[off] >> 4, seg[off] & 15
                n = 64 * (2 if pq else 1)
                vals = []
                for i in range(64):
                    if pq:
                        vals.append(struct.unpack(">H", seg[off + 1 + 2 * i:off + 3 + 2 * i])[0])
                    else:
                        vals.append(seg[off + 1 + i])
                quant[tq] = vals
                off += 1 + n
        elif marker == 0xC4:
            off = 0
            while off < len(seg):
                kind, tid = seg[off] >> 4, seg[off] & 15
                counts = list(seg[off + 1:off + 17])
                symbols = list(seg[off + 17:off + 17 + sum(counts)])
                (huff_dc if kind == 0 else huff_ac)[tid] = build_huffman(counts, symbols)
                off += 17 + sum(counts)
        elif marker in (0xC0, 0xC1):
            height, width = struct.unpack(">HH", seg[1:5])
            for i in range(seg[5]):
                cid, hv, tq = seg[6 + 3 * i:9 + 3 * i]
                comps[cid] = {"h": hv >> 4, "v": hv & 15, "tq": tq}
        elif marker == 0xC2:
            raise JpegError("progressive jpeg not supported")
        elif marker == 0xDD:
            restart_interval = struct.unpack(">H", seg[0:2])[0]
        elif marker == 0xDA:
            return decode_scan(data, pos + 2 + size, seg, comps, quant, huff_dc, huff_ac,
                               width, height, restart_interval)
        pos += 2 + size
    raise JpegError("no scan found")


def decode_scan(data, start, header, comps, quant, huff_dc, huff_ac, width, height, restart_interval):
    scan = []
    for i in range(header[0]):
        cid, tt = header[1 + 2 * i], header[2 + 2 * i]
        scan.append((cid, tt >> 4, tt & 15))
    hmax = max(comps[c]["h"] for c in comps)
    vmax = max(comps[c]["v"] for c in comps)
    mcux = (width + 8 * hmax - 1) // (8 * hmax)
    mcuy = (height + 8 * vmax - 1) // (8 * vmax)
    order = sorted(comps, key=lambda c: c)
    y_id = order[0]
    reader = BitReader(data, start)
    preds = dict((cid, 0) for cid in comps)
    y_sums = {}
    y_counts = {}
    chroma = {}
    total_mcu = mcux * mcuy
    since_restart = 0
    try:
        for mcu in range(total_mcu):
            if restart_interval and mcu and mcu % restart_interval == 0:
                # Skip to the restart marker and reset predictors.
                reader.bits = 0
                reader.buf = 0
                while reader.pos + 1 < len(data):
                    if data[reader.pos] == 0xFF and 0xD0 <= data[reader.pos + 1] <= 0xD7:
                        reader.pos += 2
                        break
                    reader.pos += 1
                preds = dict((cid, 0) for cid in comps)
            my = mcu // mcux
            for cid, dct, act in scan:
                info = comps[cid]
                blocks = info["h"] * info["v"] if len(scan) > 1 else 1
                for _ in range(blocks):
                    cat = reader.decode(huff_dc[dct])
                    preds[cid] += receive_extend(reader, cat)
                    value = preds[cid] * quant[info["tq"]][0]
                    if cid == y_id:
                        y_sums[my] = y_sums.get(my, 0) + value
                        y_counts[my] = y_counts.get(my, 0) + 1
                    else:
                        s, n = chroma.get(cid, (0, 0))
                        chroma[cid] = (s + value, n + 1)
                    # Skip the AC coefficients.
                    k = 1
                    actab = huff_ac[act]
                    while k < 64:
                        sym = reader.decode(actab)
                        run, sz = sym >> 4, sym & 15
                        if sz == 0:
                            if run != 15:
                                break
                            k += 16
                        else:
                            k += run
                            reader.get_bits(sz)
                            k += 1
            since_restart += 1
    except JpegError:
        # A scan that ends a block early (final padding) still yields its rows.
        if len(y_sums) < 4:
            raise
    rows = sorted(y_sums)
    y_rows = [y_sums[r] / max(1, y_counts[r]) / 8.0 + 128.0 for r in rows]
    cbcr = []
    ids = [c for c in order if c != y_id]
    for cid in ids[:2]:
        s, n = chroma.get(cid, (0, 0))
        cbcr.append(s / max(1, n) / 8.0 + 128.0)
    while len(cbcr) < 2:
        cbcr.append(128.0)
    return y_rows, cbcr[0], cbcr[1]


def photo_features(data):
    """(sky_rgb, ground_rgb, bright) from DC averages; raises JpegError."""
    y_rows, cb, cr = parse_jpeg_dc(data)
    if not y_rows:
        raise JpegError("no DC rows")

    def rgb(y):
        r = y + 1.402 * (cr - 128.0)
        g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
        b = y + 1.772 * (cb - 128.0)
        return (max(0.0, min(255.0, r)), max(0.0, min(255.0, g)), max(0.0, min(255.0, b)))

    third = max(1, len(y_rows) // 3)
    sky_y = sum(y_rows[:third]) / third
    ground_y = sum(y_rows[-third:]) / third
    mean_y = sum(y_rows) / len(y_rows)
    return rgb(sky_y), rgb(ground_y), mean_y, rgb(mean_y)


# Climate-band land centroids (lat, lon): land-weighted centres of each band.
CENTROIDS = {
    "snow": (61.0, 95.0),        # boreal Russia / Canada belt
    "desert": (23.5, 22.0),      # Sahara-centred arid belt
    "tropical": (3.0, 22.0),     # Congo-basin-centred tropics
    "temperate": (44.0, 28.0),   # temperate northern landmass
    "mediterranean": (37.0, 18.0),
    "southern": (-27.0, 24.0),   # southern temperate land
    "fallback": (32.0, 18.0),
}


def classify(sky, ground, bright, mean_rgb):
    r, g, b = mean_rgb
    sat = max(r, g, b) - min(r, g, b)
    green_excess = g - (r + b) / 2.0
    sand = r > 150 and g > 105 and b < 105 and r - b > 55
    if bright > 205 and sat < 38:
        return "snow"
    if sand and sky[2] >= sky[0]:
        return "desert"
    if green_excess > 14 and bright < 165:
        return "tropical" if bright < 125 else "temperate"
    if green_excess > 6:
        return "temperate"
    if sand:
        return "desert"
    if bright > 185:
        return "mediterranean"
    return "fallback"


def locate(data):
    try:
        sky, ground, bright, mean_rgb = photo_features(data)
        band = classify(sky, ground, bright, mean_rgb)
    except Exception:
        band = "fallback"
    return CENTROIDS[band]


# ---------------- the model gateway (raw vendor APIs, stdlib only) ----------------

MAX_TOKENS = 500
BANK_CONFIDENCE = 55      # at/above: answer posted straight after pass 1
SECOND_LOOKS = 10         # challenge re-looks for the least confident photos
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
RETRIES = 3
MODEL_TIMEOUT_S = 120

SYSTEM = (
    "You are an expert photo geolocation analyst (GeoGuessr world-championship level). "
    "You are shown one street-level photo from anywhere on Earth (Mapillary/OpenStreetView-5M, "
    "metadata stripped) and must estimate where it was taken. Your score is the great-circle "
    "distance from your answer to the true location, summed over photos: minimize expected distance.\n\n"
    "Examine, in order: (1) any text: language, script/alphabet, place names, phone and URL formats; "
    "(2) driving side, license plates (colours, EU band, shape), vehicle types; "
    "(3) road furniture: centreline and edge-line colours and patterns, sign shapes and colours "
    "(chevrons, kilometre markers, bollards and delineator posts and their reflector colours), "
    "guardrails, utility poles and crossarms; "
    "(4) vegetation, soil and rock colour, terrain, crops; "
    "(5) architecture: roofs, materials, fences; "
    "(6) sun position and shadows for hemisphere and latitude band; "
    "(7) people, clothing, flags and brands.\n\n"
    "Rules: commit to the single most likely location; never average two distant candidates into an "
    "ocean or empty midpoint. If you can only narrow to a country or region, answer its "
    "population-weighted centre (where its roads and towns are), not its geographic centre, and use "
    "the photo's vegetation and terrain to place yourself within it rather than at its capital by "
    "default. Longitude errors cost the same as latitude errors.\n\n"
    "Answer format: 3-6 short lines naming the decisive clues and the region they imply, then a final "
    'line containing ONLY one JSON object: {"lat": <degrees north>, "lon": <degrees east>, '
    '"confidence": <0-100>}. Confidence is your honest probability (percent) that the true location '
    "is within 200 km of your answer."
)
FIRST_LOOK = "Where was this photo taken? Follow the format exactly: brief decisive clues, then the JSON line."
SECOND_LOOK = (
    "A first estimate placed this photo at lat {lat}, lon {lon} (confidence {conf}). Re-examine the "
    "photo from scratch: name the strongest clue against that estimate and the best alternative "
    "region. Then commit to your final answer in the same format (brief clues, then the JSON line). "
    "Keep the first estimate only if the clues truly support it best."
)

# (vendor, model) tried in order; the active candidate only moves forward, on
# a 401 (that vendor has no pushed key) or model_not_allowed.
CANDIDATES = (
    ("anthropic", "claude-opus-5-5"),
    ("anthropic", "claude-sonnet-5-5"),
    ("anthropic", "claude-haiku-5-5"),
    ("openai", "gpt-6.1-sol"),
    ("openai", "gpt-6.1-terra"),
)
GATEWAY_BUDGET_TYPES = ("budget_exhausted", "request_cap")


def _dumps(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _endpoint(vendor):
    if vendor == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        return base.rstrip("/"), os.environ.get("ANTHROPIC_API_KEY") or ""
    base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return base.rstrip("/"), os.environ.get("OPENAI_API_KEY") or ""


def _request_body(vendor, model, jpeg, user_text):
    b64 = base64.b64encode(jpeg).decode("ascii")
    if vendor == "anthropic":
        return _dumps({
            "model": model,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": user_text},
            ]}],
        })
    return _dumps({
        "model": model,
        "max_completion_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": user_text},
            ]},
        ],
    })


def _error_type(raw):
    try:
        err = json.loads(raw.decode("utf-8", errors="replace")).get("error") or {}
    except ValueError:
        return None
    if isinstance(err, dict):
        for key in ("type", "code"):
            if isinstance(err.get(key), str):
                return err[key]
    return None


def _answer_text(vendor, raw):
    try:
        msg = json.loads(raw.decode("utf-8"))
    except ValueError:
        return ""
    if vendor == "anthropic":
        return "\n".join(b.get("text") or "" for b in msg.get("content") or [] if b.get("type") == "text")
    try:
        content = msg["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text") or "" for p in content if isinstance(p, dict))
    return ""


class ModelRoute:
    """Sequential gateway calls over CANDIDATES; state moves only forward.

    ask() returns (kind, text): "ok" with the model's text; "budget" when the
    trial budget/request cap is spent (no further calls are made); "failed"
    for anything else final or unreachable. A 401 or model_not_allowed
    advances to the next candidate and retries the same photo on it. Provider
    transient statuses are retried with fixed backoff unless the gateway
    marks the answer x-should-retry: false; a connection that fails before
    any answer is never retried (the gateway holds no recording to replay).
    """

    def __init__(self):
        self.index = 0
        self.budget_dead = False

    def ask(self, jpeg, user_text):
        while not self.budget_dead and self.index < len(CANDIDATES):
            vendor, model = CANDIDATES[self.index]
            kind, text = self._call(vendor, model, jpeg, user_text)
            if kind == "advance":
                self.index += 1
                continue
            if kind == "budget":
                self.budget_dead = True
                return "budget", ""
            return kind, text
        return "budget" if self.budget_dead else "failed", ""

    def _call(self, vendor, model, jpeg, user_text):
        base, key = _endpoint(vendor)
        if vendor == "anthropic":
            url = base + "/v1/messages"
            headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            if key:
                headers["x-api-key"] = key
        else:
            url = base + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = "Bearer " + key
        body = _request_body(vendor, model, jpeg, user_text)
        for attempt in range(RETRIES + 1):
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            try:
                with OPENER.open(req, timeout=MODEL_TIMEOUT_S) as res:
                    status, raw, retry = res.status, res.read(), None
            except urllib.error.HTTPError as err:
                try:
                    status, raw = err.code, err.read()
                    retry = err.headers.get("x-should-retry")
                finally:
                    err.close()
            except (urllib.error.URLError, OSError):
                return "failed", ""
            if 200 <= status < 300:
                return "ok", _answer_text(vendor, raw)
            etype = _error_type(raw)
            if etype in GATEWAY_BUDGET_TYPES:
                return "budget", ""
            if status == 401 or etype == "model_not_allowed":
                return "advance", ""
            if status in RETRY_STATUSES and retry != "false" and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            sys.stderr.write(f"v2: {vendor}/{model} HTTP {status} {etype or ''}: "
                             f"{raw[:200]!r}\n")
            return "failed", ""
        return "failed", ""


_JSON_OBJ = re.compile(r"\{[^{}]*\}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _is_lat(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and -90 <= v <= 90


def _is_lon(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and -180 <= v <= 180


def parse_guess(text):
    """(lat, lon, confidence 0-100) from a model answer, or None.

    The LAST JSON object carrying lat/lon keys wins (clues first, verdict
    last); a swapped pair that only fits exchanged is unswapped. With no
    usable JSON, the last two in-range numbers that read as a lat/lon pair.
    Missing confidence counts 0, so the photo gets a second look.
    """
    if not text:
        return None
    best = None
    for m in _JSON_OBJ.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        lat = obj.get("lat", obj.get("latitude"))
        lon = obj.get("lon", obj.get("lng", obj.get("longitude")))
        if _is_lat(lat) and _is_lon(lon):
            pass
        elif _is_lat(lon) and _is_lon(lat):
            lat, lon = lon, lat
        else:
            continue
        conf = obj.get("confidence", 0)
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            conf = 0
        conf = float(conf)
        if 0 < conf <= 1:
            conf *= 100.0
        best = (float(lat), float(lon), max(0.0, min(100.0, conf)))
    if best is not None:
        return best
    nums = [float(m.group(0)) for m in _NUMBER.finditer(text)]
    for i in range(len(nums) - 2, -1, -1):
        if _is_lat(nums[i]) and _is_lon(nums[i + 1]):
            return nums[i], nums[i + 1], 0.0
    return None


def _post(env_url, answers):
    """One POST /answers exchange: (status, parsed body or None)."""
    req = urllib.request.Request(env_url + "/answers", data=_dumps({"answers": answers}),
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(req, timeout=120) as res:
            status, raw = res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            status, raw = err.code, err.read()
        finally:
            err.close()
    except (urllib.error.URLError, OSError):
        return 0, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError:
        return status, None


def post_answers(env_url, answers):
    """Post answers, batch first then one by one; return the settled ids.

    A refused batch changes nothing on the env, so retrying singly is safe;
    a 409 singly means that photo is already answered, which also settles it.
    """
    if not answers:
        return set()
    status, body = _post(env_url, answers)
    if status == 200 and isinstance(body, dict):
        return set(body.get("accepted") or [a["id"] for a in answers])
    settled = set()
    for answer in answers:
        status, _ = _post(env_url, [answer])
        if status in (200, 409):
            settled.add(answer["id"])
    return settled


def _answer(photo_id, lat, lon):
    return {"id": photo_id, "lat": round(float(lat), 6), "lon": round(float(lon), 6)}


def main():
    env_url = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    listing = http_json(env_url + "/photos")
    todo = [p for p in listing["photos"] if not p.get("answered")]
    jpegs = {}
    for photo in todo:
        jpegs[photo["id"]] = http_bytes(env_url + photo.get("url", "/photos/" + photo["id"] + ".jpg"))

    guesses = {}   # id -> (lat, lon, confidence) or None
    banked = set()
    try:
        route = ModelRoute()
        for photo in todo:
            kind, text = route.ask(jpegs[photo["id"]], FIRST_LOOK)
            guesses[photo["id"]] = parse_guess(text) if kind == "ok" else None

        confident = [_answer(p["id"], *guesses[p["id"]][:2]) for p in todo
                     if guesses[p["id"]] and guesses[p["id"]][2] >= BANK_CONFIDENCE]
        banked = post_answers(env_url, confident)

        retry = [p for p in todo if p["id"] not in banked]
        retry.sort(key=lambda p: (guesses[p["id"]][2] if guesses[p["id"]] else -1.0,
                                  todo.index(p)))
        for photo in retry[:SECOND_LOOKS]:
            first = guesses[photo["id"]]
            if first is None:
                prompt = FIRST_LOOK
            else:
                prompt = SECOND_LOOK.format(lat=f"{first[0]:.4f}", lon=f"{first[1]:.4f}",
                                            conf=f"{first[2]:.0f}")
            kind, text = route.ask(jpegs[photo["id"]], prompt)
            second = parse_guess(text) if kind == "ok" else None
            if second is not None and (first is None or second[2] >= first[2]):
                guesses[photo["id"]] = second
    except Exception as err:  # banking below uses whatever guesses exist
        sys.stderr.write(f"v2: model phase stopped early: {err!r}\n")

    final = []
    for photo in todo:
        if photo["id"] in banked:
            continue
        guess = guesses.get(photo["id"])
        lat, lon = guess[:2] if guess else locate(jpegs[photo["id"]])
        final.append(_answer(photo["id"], lat, lon))
    banked |= post_answers(env_url, final)

    for photo in todo:  # last resort: no photo stays unanswered
        if photo["id"] not in banked:
            lat, lon = locate(jpegs[photo["id"]])
            banked |= post_answers(env_url, [_answer(photo["id"], lat, lon)])


if __name__ == "__main__":
    main()
