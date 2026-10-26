#!/usr/bin/env python3
"""
convert.py - turn a Thai Wikipedia SQL dump into a PageRank-ready graph.

Input  (from https://dumps.wikimedia.org/thwiki/latest/):
    thwiki-latest-page.sql.gz        every page: id, namespace, title, is_redirect
    thwiki-latest-redirect.sql.gz    redirect page -> target title
    thwiki-latest-pagelinks.sql.gz   link: from page -> target
    thwiki-latest-linktarget.sql.gz  (newer dumps only) target id -> title

Output (in --out directory):
    graph_100.bin    full graph in CSR format, incoming links (see write_graph)
    graph_50.bin, graph_25.bin, graph_10.bin   smaller graphs for the speedup-vs-size chart
    titles.txt       title of vertex i on line i (UTF-8, Thai)

Usage:
    python3 convert.py --dump DIR --out DIR
"""
import argparse, gzip, os, re, struct, sys, time
from array import array
import numpy as np

# ---------------------------------------------------------------------------
# 1. Reading MySQL dump files
# ---------------------------------------------------------------------------
# A dump is a .sql.gz file containing:
#   CREATE TABLE `page` ( `page_id` int ..., `page_namespace` int ..., ... );
#   INSERT INTO `page` VALUES (1,0,'Title',0,...),(2,0,'Other',1,...),...;
# We read the CREATE TABLE to learn the column order (so we do not depend on
# one fixed schema), then stream the INSERT lines tuple by tuple.

VALUE_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?|NULL|'(?:[^'\\]|\\.)*'", re.S)
TUPLE_RE = re.compile(
    r"\(((?:-?\d+(?:\.\d+)?(?:e-?\d+)?|NULL|'(?:[^'\\]|\\.)*')"
    r"(?:,(?:-?\d+(?:\.\d+)?(?:e-?\d+)?|NULL|'(?:[^'\\]|\\.)*'))*)\)", re.S)
ESCAPES = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a",
           "\\": "\\", "'": "'", '"': '"'}

def unquote(v):
    """Turn one SQL value into a Python int, str or None."""
    if v == "NULL":
        return None
    if v[0] == "'":
        s = v[1:-1]
        if "\\" in s:
            s = re.sub(r"\\(.)", lambda m: ESCAPES.get(m.group(1), m.group(1)), s)
        return s
    return int(v) if re.fullmatch(r"-?\d+", v) else float(v)

def read_table(path, wanted):
    """Yield dicts {column: value} for the columns in `wanted`, row by row."""
    columns, idx = None, None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        in_create = False
        for line in f:
            if line.startswith("CREATE TABLE"):
                in_create, columns = True, []
                continue
            if in_create:
                m = re.match(r"\s*`(\w+)`", line)
                if m:
                    columns.append(m.group(1))
                if line.startswith(")"):
                    in_create = False
                    missing = [c for c in wanted if c not in columns]
                    if missing:
                        raise SystemExit(f"{path}: columns {missing} not found; has {columns}")
                    idx = [columns.index(c) for c in wanted]
                continue
            # Old dumps put a whole INSERT (thousands of rows) on one line;
            # newer dumps write "INSERT INTO ... VALUES" and then one "(row)," per line.
            if not (line.startswith("INSERT INTO") or line.startswith("(")):
                continue
            for t in TUPLE_RE.finditer(line):
                vals = VALUE_RE.findall(t.group(1))
                yield {c: unquote(vals[i]) for c, i in zip(wanted, idx)}

def columns_of(path):
    """Return the column names declared in a dump's CREATE TABLE."""
    cols = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        inside = False
        for line in f:
            if line.startswith("CREATE TABLE"):
                inside = True
                continue
            if inside:
                m = re.match(r"\s*`(\w+)`", line)
                if m:
                    cols.append(m.group(1))
                if line.startswith(")"):
                    return cols
    return cols

# ---------------------------------------------------------------------------
# 2. Building the graph
# ---------------------------------------------------------------------------
def build(dump_dir, prefix):
    p = lambda name: os.path.join(dump_dir, f"{prefix}-{name}.sql.gz")
    t0 = time.time()

    # 2a. Pages: keep namespace 0 (real articles), remember which are redirects.
    title_to_id, is_redirect, id_to_title = {}, set(), {}
    for r in read_table(p("page"), ["page_id", "page_namespace", "page_title", "page_is_redirect"]):
        if r["page_namespace"] != 0:
            continue
        pid, title = r["page_id"], r["page_title"]
        title_to_id[title] = pid
        id_to_title[pid] = title
        if r["page_is_redirect"]:
            is_redirect.add(pid)
    print(f"pages (namespace 0): {len(id_to_title):,}  redirects: {len(is_redirect):,}  [{time.time()-t0:.0f}s]")

    # 2b. Redirects: redirect page id -> final article id (follow one hop).
    redirect_to = {}
    for r in read_table(p("redirect"), ["rd_from", "rd_namespace", "rd_title"]):
        if r["rd_from"] in is_redirect and r["rd_namespace"] == 0:
            tgt = title_to_id.get(r["rd_title"])
            if tgt is not None and tgt not in is_redirect:
                redirect_to[r["rd_from"]] = tgt

    def resolve(pid):
        """Article id a link really points to, or None (missing / broken redirect)."""
        if pid is None:
            return None
        if pid in is_redirect:
            return redirect_to.get(pid)
        return pid

    # 2c. Articles = namespace-0 pages that are not redirects. Number them 0..V-1.
    articles = sorted(pid for pid in id_to_title if pid not in is_redirect)
    vid = {pid: i for i, pid in enumerate(articles)}
    V = len(articles)

    # 2d. Links. Newer dumps store the target as pl_target_id -> linktarget table;
    #     older dumps store pl_namespace + pl_title directly. Support both.
    pl_cols = columns_of(p("pagelinks"))
    src, dst = array("i"), array("i")      # compact int32 arrays, not Python lists
    if "pl_target_id" in pl_cols:
        target_of = {}  # linktarget id -> article vertex
        for r in read_table(p("linktarget"), ["lt_id", "lt_namespace", "lt_title"]):
            if r["lt_namespace"] == 0:
                a = resolve(title_to_id.get(r["lt_title"]))
                if a is not None:
                    target_of[r["lt_id"]] = vid[a]
        rows = read_table(p("pagelinks"), ["pl_from", "pl_from_namespace", "pl_target_id"])
        get_target = lambda r: target_of.get(r["pl_target_id"])
    else:
        rows = read_table(p("pagelinks"), ["pl_from", "pl_from_namespace", "pl_namespace", "pl_title"])
        def get_target(r):
            if r["pl_namespace"] != 0:
                return None
            a = resolve(title_to_id.get(r["pl_title"]))
            return None if a is None else vid[a]

    for r in rows:
        if r["pl_from_namespace"] != 0:
            continue
        u = vid.get(r["pl_from"])          # source must be a real article
        if u is None:
            continue
        v = get_target(r)
        if v is None or v == u:            # drop missing targets and self-links
            continue
        src.append(u); dst.append(v)

    src = np.frombuffer(src, dtype=np.int32).astype(np.int64)
    dst = np.frombuffer(dst, dtype=np.int32).astype(np.int64)
    # Remove duplicate links (same page linked twice, or via two redirects).
    key = np.unique(src * V + dst)
    src, dst = (key // V).astype(np.int32), (key % V).astype(np.int32)
    titles = [id_to_title[pid] for pid in articles]
    print(f"articles V = {V:,}  links E = {len(src):,}  [{time.time()-t0:.0f}s]")
    return V, src, dst, titles

# ---------------------------------------------------------------------------
# 3. CSR of incoming links + writing the binary file
# ---------------------------------------------------------------------------
def write_graph(path, V, src, dst):
    """
    Binary layout (little-endian):
        int32 V, int32 E
        int32 row_ptr[V+1]   incoming links of v are col_idx[row_ptr[v] .. row_ptr[v+1]-1]
        int32 col_idx[E]     the source page of each incoming link
        int32 out_deg[V]     number of outgoing links of each page (0 = dangling)
    """
    order = np.argsort(dst, kind="stable")          # group links by destination
    col_idx = src[order].astype(np.int32)
    counts = np.bincount(dst, minlength=V)
    row_ptr = np.zeros(V + 1, dtype=np.int32)
    np.cumsum(counts, out=row_ptr[1:])
    out_deg = np.bincount(src, minlength=V).astype(np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", V, len(col_idx)))
        f.write(row_ptr.tobytes()); f.write(col_idx.tobytes()); f.write(out_deg.tobytes())
    dangling = int((out_deg == 0).sum())
    print(f"  {os.path.basename(path)}: V={V:,} E={len(col_idx):,} dangling={dangling:,} "
          f"max in-degree={int(counts.max()) if V else 0:,}")

def subgraph(V, src, dst, frac, seed=42):
    """Induced subgraph on a random `frac` of the vertices (same seed = same result)."""
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(V, size=max(1, int(V * frac)), replace=False))
    new_id = np.full(V, -1, dtype=np.int64); new_id[keep] = np.arange(len(keep))
    m = (new_id[src] >= 0) & (new_id[dst] >= 0)
    return len(keep), new_id[src[m]].astype(np.int32), new_id[dst[m]].astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="folder with the .sql.gz files")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--prefix", default="thwiki-latest")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    V, src, dst, titles = build(a.dump, a.prefix)
    with open(os.path.join(a.out, "titles.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(t.replace("_", " ") for t in titles) + "\n")
    write_graph(os.path.join(a.out, "graph_100.bin"), V, src, dst)
    for pct in (50, 25, 10):
        sv, ss, sd = subgraph(V, src, dst, pct / 100)
        write_graph(os.path.join(a.out, f"graph_{pct}.bin"), sv, ss, sd)

if __name__ == "__main__":
    main()
