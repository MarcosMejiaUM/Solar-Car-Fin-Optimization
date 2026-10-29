"""
finopt.airfoil_db — fetch airfoil coordinate files by name, with caching.

D1 of the roadmap. Turns a human name ("e327", "ch10", "naca2412") into a
cached local .dat file the pipeline can load, so the cross-family comparison
doesn't depend on manually hunting coordinate files.

Sources, tried in order:
  1. airfoiltools.com  — has a clean Selig-format endpoint and consistent
     "<name>-il" identifiers; most reliable.
  2. UIUC m-selig database — the canonical source, but naming is inconsistent
     (ch10sm vs ch10 etc.), so it's the fallback.

Downloaded files are cached under airfoils/ and never re-fetched. If both
sources miss, the error lists every URL tried AND tells you to drop the .dat
in airfoils/ manually — the loader will pick it up with no network at all.

Nothing here is solar-car-specific; it's a general airfoil fetcher.
"""

from __future__ import annotations

import os
import urllib.request
import urllib.error

# logical name -> list of candidate identifiers to try at each source.
# Add aliases here as you discover the exact catalogue names.
_ALIASES = {
    "e327":     ["e327", "e327-il"],
    "eppler327": ["e327", "e327-il"],
    "ch10":     ["ch10sm", "ch10sm-il", "ch10", "ch10-il"],
    "ch10sm":   ["ch10sm", "ch10sm-il"],
}

_AIRFOILTOOLS = "http://airfoiltools.com/airfoil/seligdatfile?airfoil={id}"
_UIUC = [
    "https://m-selig.ae.illinois.edu/ads/coord_seligFmt/{id}.dat",
    "https://m-selig.ae.illinois.edu/ads/coord/{id}.dat",
]

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "airfoils")


def _candidates(name: str):
    """Identifiers to try for a logical name."""
    key = name.lower().replace(" ", "").replace("-", "")
    if key in _ALIASES:
        return _ALIASES[key]
    # unknown name: try it bare and with the airfoiltools -il suffix
    base = name.lower().replace(" ", "")
    return [base, f"{base}-il"]


def _looks_like_coords(text: str) -> bool:
    """Cheap sanity check that a download is actually a coordinate file."""
    n = 0
    for ln in text.splitlines():
        p = ln.split()
        if len(p) == 2:
            try:
                float(p[0]); float(p[1]); n += 1
            except ValueError:
                pass
    return n >= 10


def fetch_airfoil(name: str, cache_dir: str = None,
                  force: bool = False, timeout: float = 30.0) -> str:
    """Return a local path to the airfoil's .dat file, downloading if needed.

    Parameters
    ----------
    name : logical name ("e327", "ch10", "naca2412", or any airfoiltools id).
    cache_dir : where to store/look for files (default: project airfoils/).
    force : re-download even if cached.
    timeout : per-request seconds.

    Raises
    ------
    RuntimeError if the airfoil could not be fetched from any source, with a
    message listing every URL tried and how to supply the file manually.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)

    # 0) already cached? (match on logical name OR any candidate id)
    if not force:
        for stem in [name.lower().replace(" ", "")] + _candidates(name):
            p = os.path.join(cache_dir, f"{stem}.dat")
            if os.path.isfile(p):
                return p

    tried = []
    for ident in _candidates(name):
        urls = [_AIRFOILTOOLS.format(id=ident)] + \
               [u.format(id=ident) for u in _UIUC]
        for url in urls:
            tried.append(url)
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "finopt-airfoil-fetch/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, urllib.error.HTTPError, Exception):
                continue
            if _looks_like_coords(text):
                out = os.path.join(cache_dir, f"{name.lower().replace(' ','')}.dat")
                with open(out, "w") as f:
                    f.write(text)
                return out

    raise RuntimeError(
        f"Could not fetch airfoil '{name}'. Tried:\n  " +
        "\n  ".join(tried) +
        f"\n\nFix: download the Selig-format .dat yourself (airfoiltools.com "
        f"or the UIUC database) and save it as "
        f"{os.path.join(cache_dir, name.lower().replace(' ','') + '.dat')} "
        f"— the loader will use it with no network needed. If you know the "
        f"catalogue id, add it to _ALIASES in airfoil_db.py.")


def ensure_airfoils(names, cache_dir: str = None) -> dict:
    """Fetch several airfoils; return {name: path}. Reports misses without
    aborting the whole batch (so one bad name doesn't stop the others)."""
    out = {}
    for nm in names:
        try:
            out[nm] = fetch_airfoil(nm, cache_dir=cache_dir)
        except RuntimeError as e:
            out[nm] = None
            print(f"[fetch] {nm}: FAILED\n{e}\n")
    return out
