"""photo-geolocator-01 entry: vision-model geolocator, code fallback.

Each unanswered photo is shown to a vision model reached through the
gateway (Anthropic Messages API first, OpenAI Chat Completions as the
spare vendor). The model answers with clue lines and one JSON verdict
giving lat, lon and a 0-100 confidence. Answers at or above 55 are
posted straight away; the ten least confident leftovers get one
challenge second look that argues against the first estimate. Photos
the model route cannot place (refusal, empty or unparseable reply,
spent budget) fall back to the code-only locator kept from v1: decode
only the DC coefficients of the JPEG, average sky and ground rows in
YCbCr, and map the colour climate (snow, desert sand, lush tropics,
temperate greenery, dark boreal forest) to that band's land centroid.
Every photo ends answered. Requests are a pure function of the photo
bytes, the fixed prompts and earlier replies, sent strictly in photo
order, one call at a time, so only one worst-case reservation is ever
held against the $1.00 trial budget.
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


TOKEN_CAP = 500
BANK_AT = 55
CHALLENGE_COUNT = 10
TRANSIENT = (408, 429, 500, 502, 503, 504, 529)
MAX_RETRIES = 3
CALL_TIMEOUT = 120

ANALYST_BRIEF = (
    "You are a specialist in photo geolocation. Your task is to pin down "
    "where one street-level photograph was taken. It may come from "
    "anywhere on Earth and all metadata has been removed. The score for "
    "each photo is the great-circle distance in kilometres between the "
    "point you give and the true point, added up across photos, so aim "
    "for the point that minimises the distance you expect to be out by. "
    "Work through the evidence in this sequence: first anything readable "
    "(language, alphabet or script, place names, telephone and web "
    "formats); then which side of the road traffic uses and what plates "
    "look like (colours, bands, proportions, vehicle mix); then road "
    "equipment (centre and edge line colours and patterns, sign shapes "
    "and colours, chevron boards, distance markers, bollards, delineator "
    "posts including the colour of their reflectors, barriers, power "
    "poles and crossarm styles); then plants, soil and rock colour, land "
    "shape and crops; then buildings (roof forms, wall materials, "
    "fences); then the sun and shadows to fix the hemisphere and rough "
    "latitude band; then people, dress, flags and brands. Always commit "
    "to one best point: never split the difference between two far-apart "
    "candidates into an ocean or empty midpoint. When you can only place "
    "the photo inside a country or region, answer where its people and "
    "roads concentrate (its population-weighted centre) and use the "
    "vegetation and terrain in shot to position yourself inside it "
    "rather than defaulting to the capital. Reply with three to six "
    "short lines naming the decisive clues and the area they point to, "
    "then a final line holding only a JSON object of the form {\"lat\": "
    "number, \"lon\": number, \"confidence\": number} with lat in degrees "
    "north, lon in degrees east and confidence from 0 to 100 meaning "
    "your honest percentage chance that the truth lies within 200 km "
    "of the point you gave."
)
ASK_FIRST = (
    "Where was this photograph taken? Follow the required reply shape "
    "exactly: brief decisive clues first, then the JSON line."
)
ASK_CHALLENGE = (
    "An earlier estimate put this photograph at latitude {lat}, "
    "longitude {lon} with confidence {conf}. Look again from the start: "
    "state the strongest clue that tells against that estimate and the "
    "best rival area, then commit to your final answer in the same shape "
    "(brief clues, then the JSON line). Keep the earlier point only if "
    "the evidence really favours it over every rival."
)

ROUTE = (
    ("anthropic", "claude-opus-5-5"),
    ("anthropic", "claude-sonnet-5-5"),
    ("anthropic", "claude-haiku-5-5"),
    ("openai", "gpt-6.1-sol"),
    ("openai", "gpt-6.1-terra"),
)
STOP_TYPES = ("budget_exhausted", "request_cap")


def pack(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def vendor_base(vendor):
    if vendor == "anthropic":
        return (os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/"), os.environ.get("ANTHROPIC_API_KEY") or ""
    return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"), os.environ.get("OPENAI_API_KEY") or ""


def make_payload(vendor, model, jpeg, user_text):
    encoded = base64.b64encode(jpeg).decode("ascii")
    if vendor == "anthropic":
        return pack({
            "model": model,
            "max_tokens": TOKEN_CAP,
            "temperature": 0,
            "system": ANALYST_BRIEF,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}},
                {"type": "text", "text": user_text},
            ]}],
        })
    return pack({
        "model": model,
        "max_completion_tokens": TOKEN_CAP,
        "messages": [
            {"role": "system", "content": ANALYST_BRIEF},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
                {"type": "text", "text": user_text},
            ]},
        ],
    })


def reported_error(raw):
    try:
        node = json.loads(raw.decode("utf-8", errors="replace")).get("error")
    except ValueError:
        return None
    if isinstance(node, dict):
        for field in ("type", "code"):
            val = node.get(field)
            if isinstance(val, str):
                return val
    return None


def reply_text(vendor, raw):
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError:
        return ""
    if vendor == "anthropic":
        return "\n".join(str(b.get("text") or "") for b in doc.get("content") or [] if isinstance(b, dict) and b.get("type") == "text")
    try:
        content = doc["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return ""


class Gateway:
    def __init__(self):
        self.pos = 0
        self.spent = False

    def ask(self, jpeg, user_text):
        while not self.spent and self.pos < len(ROUTE):
            vendor, model = ROUTE[self.pos]
            outcome, text = self._try(vendor, model, jpeg, user_text)
            if outcome == "next":
                self.pos += 1
                continue
            if outcome == "stop":
                self.spent = True
                return "budget", ""
            return outcome, text
        return ("budget", "") if self.spent else ("failed", "")

    def _try(self, vendor, model, jpeg, user_text):
        base, key = vendor_base(vendor)
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
        body = make_payload(vendor, model, jpeg, user_text)
        for attempt in range(MAX_RETRIES + 1):
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            try:
                with OPENER.open(req, timeout=CALL_TIMEOUT) as res:
                    status, raw, hint = res.status, res.read(), None
            except urllib.error.HTTPError as err:
                try:
                    status, raw = err.code, err.read()
                    hint = err.headers.get("x-should-retry")
                finally:
                    err.close()
            except (urllib.error.URLError, OSError):
                return "failed", ""
            if 200 <= status < 300:
                return "ok", reply_text(vendor, raw)
            kind = reported_error(raw)
            if kind in STOP_TYPES:
                return "stop", ""
            if status == 401 or kind == "model_not_allowed":
                return "next", ""
            if status in TRANSIENT and hint != "false" and attempt < MAX_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            sys.stderr.write("geo: %s/%s HTTP %s %s: %r\n" % (vendor, model, status, kind or "", raw[:200]))
            return "failed", ""
        return "failed", ""


BRACED = re.compile(r"\{[^{}]*\}")
PLAIN_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def lat_ok(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and -90 <= v <= 90


def lon_ok(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and -180 <= v <= 180


def read_verdict(text):
    if not text:
        return None
    chosen = None
    for match in BRACED.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        lat = obj.get("lat", obj.get("latitude"))
        lon = obj.get("lon", obj.get("lng", obj.get("longitude")))
        if lat_ok(lat) and lon_ok(lon):
            pass
        elif lat_ok(lon) and lon_ok(lat):
            lat, lon = lon, lat
        else:
            continue
        conf = obj.get("confidence", 0)
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            conf = 0
        conf = float(conf)
        if 0 < conf <= 1:
            conf = conf * 100.0
        chosen = (float(lat), float(lon), max(0.0, min(100.0, conf)))
    if chosen is not None:
        return chosen
    nums = [float(m.group(0)) for m in PLAIN_NUM.finditer(text)]
    i = len(nums) - 2
    while i >= 0:
        if lat_ok(nums[i]) and lon_ok(nums[i + 1]):
            return (nums[i], nums[i + 1], 0.0)
        i -= 1
    return None


def send_once(env_url, batch):
    req = urllib.request.Request(env_url + "/answers", data=pack({"answers": batch}), method="POST", headers={"Content-Type": "application/json"})
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


def deliver(env_url, batch):
    if not batch:
        return set()
    status, body = send_once(env_url, batch)
    if status == 200 and isinstance(body, dict):
        return set(body.get("accepted") or [a["id"] for a in batch])
    done = set()
    for item in batch:
        st, _ignored = send_once(env_url, [item])
        if st in (200, 409):
            done.add(item["id"])
    return done


def entry(photo_id, lat, lon):
    return {"id": photo_id, "lat": round(float(lat), 6), "lon": round(float(lon), 6)}


def main():
    env_url = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    listing = http_json(env_url + "/photos")
    queue = [p for p in listing["photos"] if not p.get("answered")]
    blobs = {}
    for photo in queue:
        blobs[photo["id"]] = http_bytes(env_url + photo.get("url", "/photos/" + photo["id"] + ".jpg"))
    order = {}
    for idx, photo in enumerate(queue):
        order[photo["id"]] = idx
    verdicts = {}
    settled = set()
    try:
        gate = Gateway()
        for photo in queue:
            state, text = gate.ask(blobs[photo["id"]], ASK_FIRST)
            verdicts[photo["id"]] = read_verdict(text) if state == "ok" else None
        sure = [entry(p["id"], verdicts[p["id"]][0], verdicts[p["id"]][1]) for p in queue if verdicts[p["id"]] and verdicts[p["id"]][2] >= BANK_AT]
        settled = deliver(env_url, sure)
        pending = [p for p in queue if p["id"] not in settled]

        def rank(p):
            v = verdicts[p["id"]]
            return (v[2] if v else -1.0, order[p["id"]])

        pending.sort(key=rank)
        for photo in pending[:CHALLENGE_COUNT]:
            before = verdicts[photo["id"]]
            if before is None:
                prompt = ASK_FIRST
            else:
                prompt = ASK_CHALLENGE.format(lat="%.4f" % before[0], lon="%.4f" % before[1], conf="%.0f" % before[2])
            state, text = gate.ask(blobs[photo["id"]], prompt)
            after = read_verdict(text) if state == "ok" else None
            if after is not None and (before is None or after[2] >= before[2]):
                verdicts[photo["id"]] = after
    except Exception as exc:
        sys.stderr.write("geo: model phase cut short: %r\n" % (exc,))
    rest = []
    for photo in queue:
        if photo["id"] in settled:
            continue
        v = verdicts.get(photo["id"])
        lat, lon = (v[0], v[1]) if v else locate(blobs[photo["id"]])
        rest.append(entry(photo["id"], lat, lon))
    settled |= deliver(env_url, rest)
    for photo in queue:
        if photo["id"] not in settled:
            lat, lon = locate(blobs[photo["id"]])
            settled |= deliver(env_url, [entry(photo["id"], lat, lon)])


if __name__ == "__main__":
    main()
