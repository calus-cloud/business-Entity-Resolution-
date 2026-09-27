"""Text normalization for business names and addresses.

All functions are country-agnostic: they transliterate any script to ASCII
(unidecode), canonicalize common abbreviations, and derive tokens used both for
blocking and for pairwise features.
"""
import re

from unidecode import unidecode

# Legal-form / filler words removed from the "core" business name.
LEGAL = {
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "plc", "lp", "pc", "pllc", "sarl", "sas", "sa",
    "eurl", "sasu", "gmbh", "lc", "opc",
}
NOISE = {
    "center", "centre", "services", "service", "partners", "the", "and", "of", "et",
    "mr", "mrs", "ms", "dr", "smt", "shri", "sri", "esq", "dds", "od", "md", "m",
    "s", "www", "com", "net", "org", "in", "fr", "le", "la", "les", "de", "du", "des",
}
NAME_STOP = LEGAL | NOISE

_DBA_RE = re.compile(r"\b(?:d\s*\.?\s*b\s*\.?\s*a\b\.?|a\s*\.?\s*k\s*\.?\s*a\b\.?|f\s*/\s*k\s*/\s*a\b|fka\b|formerly\b)")
_DOMAIN_RE = re.compile(r"\b(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co|fr|biz|info|co\.in)\b")
_STORE_RE = re.compile(r"#\s*\d+|\b\d{7,}\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NULL_RE = re.compile(r"<?\bnull\b>?|\bnone\b|\bn/a\b")

# Address canonicalization (token -> canonical token).
ADDR_MAP = {
    "street": "st", "saint": "st", "str": "st", "avenue": "ave", "av": "ave",
    "road": "rd", "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "trail": "trl", "boulevard": "blvd", "bd": "blvd", "place": "pl", "terrace": "ter",
    "parkway": "pkwy", "highway": "hwy", "suite": "ste", "floor": "fl", "apartment": "apt",
    "apartments": "apt", "apts": "apt", "square": "sq", "north": "n", "south": "s",
    "east": "e", "west": "w", "mount": "mt", "fort": "ft", "point": "pt", "cove": "cv",
    "crossing": "xing", "heights": "hts", "junction": "jct", "center": "ctr",
    "centre": "ctr", "township": "", "townshiip": "", "city": "", "ciyt": "",
    "number": "no", "door": "", "hno": "", "plot": "", "flat": "", "shop": "",
    "house": "", "bengaluru": "bangalore", "bombay": "mumbai", "baroda": "vadodara",
    "gurugram": "gurgaon", "first": "1", "second": "2", "third": "3", "fourth": "4",
    "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9",
    "tenth": "10", "eleventh": "11", "twelfth": "12",
}
REGIONS = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "chandigarh": "ch", "puducherry": "py", "jammu and kashmir": "jk",
}
# Longest names first so "west virginia" wins over "virginia".
_REGION_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, REGIONS), key=len, reverse=True)) + r")\b")
# Words skipped when picking the street word that follows a house number.
ADDR_SKIP = {"st", "ave", "rd", "dr", "ln", "ct", "cir", "trl", "blvd", "pl", "ter", "pkwy",
             "hwy", "ste", "fl", "apt", "no", "n", "s", "e", "w", "rue", "du", "de", "la",
             "le", "des", "avenue", "bis", "ter", "unit", "pmb", "po", "box", "a", "b", "c", "d"}


def skeleton(tok):
    """Phonetic-ish consonant skeleton; makes transliterations comparable.

    e.g. 'private' and 'praaivett' both -> 'prvt'; 'raj' and 'raaj' -> 'rj'.
    """
    if tok.isdigit():
        return tok
    t = tok.replace("ph", "f")
    t = re.sub(r"([bcdgkpstj])h", r"\1", t)
    t = t.translate(str.maketrans("wqcz", "vkkj"))
    t = re.sub(r"[aeiouy]", "", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return t or tok[:1]


LEGAL_SKEL = {skeleton(w) for w in LEGAL | {"services", "service", "partners", "center"}
              if len(skeleton(w)) >= 3} | {"prvt", "lmtd", "llp", "lp"}


def _collapse_initials(toks):
    """Join runs of single letters: ['l','l','c'] -> ['llc']."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def ascii_lower(s):
    """Transliterate to ASCII and lowercase; '&'/'+' become 'and'."""
    if not s:
        return ""
    s = unidecode(s).lower()
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"'s\b", "s", s).replace("'", "")
    return s


def normalize_name(raw):
    """Return (clean_name, core_tokens, is_domain).

    clean_name: all tokens joined; core_tokens: tokens without legal/noise words.
    """
    s = ascii_lower(raw)
    parts = _DBA_RE.split(s)
    if len(parts) > 1 and parts[-1].strip():
        s = parts[-1]  # keep the trade name after d.b.a. / aka / f/k/a
    is_domain = 0
    m = _DOMAIN_RE.search(s)
    if m:
        s = m.group(1).replace("-", " ")
        is_domain = 1
    s = _STORE_RE.sub(" ", s)
    toks = _collapse_initials(_TOKEN_RE.findall(s))
    core = [t for t in toks if t not in NAME_STOP and skeleton(t) not in LEGAL_SKEL]
    if not core:
        core = [t for t in toks if t not in LEGAL] or toks
    return " ".join(toks), core, is_domain


def normalize_address(raw):
    """Return (clean_address, word_tokens, number_tokens, house_keys)."""
    s = ascii_lower(raw)
    s = _NULL_RE.sub(" ", s)
    s = _REGION_RE.sub(lambda m: " " + REGIONS[m.group(1)] + " ", s)
    raw_toks = _TOKEN_RE.findall(s)
    words, nums, seq = [], [], []
    for t in raw_toks:
        if any(ch.isdigit() for ch in t):
            for d in re.findall(r"\d+", t):
                d = d.lstrip("0") or "0"
                nums.append(d)
                seq.append(("n", d))
            for a in re.findall(r"[a-z]+", t):
                if a not in ("st", "nd", "rd", "th") and len(a) > 1:
                    words.append(a)
                    seq.append(("w", a))
            continue
        t = ADDR_MAP.get(t, t)
        if t:
            words.append(t)
            seq.append(("w", t))
    # house key: number followed by the next informative street word
    keys = []
    for i, (kind, v) in enumerate(seq):
        if kind != "n":
            continue
        for kind2, v2 in seq[i + 1:i + 4]:
            if kind2 == "w" and v2 not in ADDR_SKIP and len(v2) > 1:
                keys.append(v + "_" + v2)
                break
    clean = " ".join(v for _, v in seq)
    return clean, words, nums, keys


def record_fields(name, address):
    """All derived fields for one record (used by preprocessing)."""
    nclean, core, is_dom = normalize_name(name)
    aclean, words, nums, keys = normalize_address(address)
    core_str = " ".join(core)
    skel = " ".join(skeleton(t) for t in core)
    concat = "".join(core)
    btoks = set()
    for t in core:
        if len(t) > 1:
            btoks.add("n:" + t)
        sk = skeleton(t)
        if len(sk) > 1:
            btoks.add("k:" + sk)
    if len(concat) >= 6:
        btoks.add("c:" + concat)
    for w in words:
        if len(w) > 1:
            btoks.add("a:" + w)
    for d in nums:
        btoks.add("d:" + d)
    for k in keys:
        btoks.add("p:" + k)
    return {
        "name_clean": nclean,
        "name_core": core_str,
        "name_skel": skel,
        "name_concat": concat,
        "is_domain": is_dom,
        "addr_clean": aclean,
        "addr_nums": " ".join(nums),
        "addr_keys": " ".join(keys),
        "btoks": " ".join(sorted(btoks)),
    }
