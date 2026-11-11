"""photo-geolocator-01 entry: code-only geolocator (no model calls).

For each photo it decodes only the DC coefficients of the JPEG (the average
colour of every 8x8 block, no trained weights, no photo table), averages the
top rows (sky) and bottom rows (ground) in YCbCr, converts the averages to
RGB, and maps the colour climate (snow, desert sand, lush tropics, temperate
greenery, dark boreal forest) to the land centroid of that climate band.
Every photo is answered, in photo order, in one POST. Deterministic: the
output is a pure function of the photo bytes.
Run: python3 -P -s /bot/main.py  (MAKEFUN_ENV_URL names the env)
"""
import json
import os
import struct
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


def main():
    env_url = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    listing = http_json(env_url + "/photos")
    answers = []
    for photo in listing["photos"]:
        if photo.get("answered"):
            continue
        jpeg = http_bytes(env_url + photo.get("url", "/photos/" + photo["id"] + ".jpg"))
        lat, lon = locate(jpeg)
        answers.append({"id": photo["id"], "lat": round(lat, 6), "lon": round(lon, 6)})
    if answers:
        http_json(env_url + "/answers", {"answers": answers})


if __name__ == "__main__":
    main()
