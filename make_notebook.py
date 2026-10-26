#!/usr/bin/env python3
"""Build pagerank_thwiki.ipynb from the current source files (so the notebook
always contains the same code as the repo). Run: python3 make_notebook.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
read = lambda p: open(os.path.join(HERE, p), encoding="utf-8").read()

cells = []
def md(text):   cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})
def code(text): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                              "outputs": [], "source": text.strip("\n")})

md("""
# PageRank on the Thai Wikipedia link graph: CPU vs GPU (CUDA)

**Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

1. Check the GPU
2. (Optional) Google Drive cache, so the dump is downloaded only once
3. Download the Thai Wikipedia dump and convert it to a graph (`prep/convert.py`)
4. Compile `pagerank.cu` with `nvcc`
5. Correctness run on the full graph and the top-20 Thai pages
6. Benchmark CPU vs three GPU versions on 10 / 25 / 50 / 100 % of the graph
7. Charts for the slides
""")

code("!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv\n!nvcc --version | tail -2")

md("## 1. Source files (same as the GitHub repo)")
code("!mkdir -p prep data dump results")
code("%%writefile prep/convert.py\n" + read("prep/convert.py"))
code("%%writefile pagerank.cu\n" + read("pagerank.cu"))

md("""
## 2. Optional: cache data in Google Drive
Colab deletes files when the session ends. Set `USE_DRIVE = True` to keep the
converted graph in `MyDrive/pagerank-thwiki/` and skip the download next time.
""")
code('''
USE_DRIVE = False
import os, shutil
CACHE = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE = '/content/drive/MyDrive/pagerank-thwiki'
    os.makedirs(CACHE + '/data', exist_ok=True)
    for f in os.listdir(CACHE + '/data'):
        shutil.copy(f'{CACHE}/data/{f}', 'data/')
print('cached files:', sorted(os.listdir('data')))
''')

md("""
## 3. Download the dump and build the graph
`linktarget` only exists in newer dumps; if it fails to download, the script
uses the older pagelinks format automatically.
""")
code('''
import os
if not os.path.exists('data/graph_100.bin'):
    for t in ['page', 'redirect', 'pagelinks', 'linktarget']:
        os.system(f'wget -q -nc -P dump https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-{t}.sql.gz')
    !ls -lh dump
    !python3 prep/convert.py --dump dump --out data
    if CACHE:
        !cp data/* {CACHE}/data/
else:
    print('graph already built')
!ls -lh data
''')

md("""
## 4. Compile
`-arch=sm_75` matches the T4 (double-precision `atomicAdd` needs sm_60 or newer).
""")
code("!nvcc -O3 -arch=sm_75 -o pagerank pagerank.cu && echo compiled")

md("""
## 5. Correctness: full graph, run until converged
Checks every GPU version against the CPU: max relative error per page and the
same top-20 pages. Then prints the 20 highest-ranked Thai Wikipedia articles.
""")
code("!./pagerank data/graph_100.bin data/titles.txt | tee results/correctness.txt")

md("""
## 6. Benchmark
Fixed 50 iterations so CPU and GPU do exactly the same work. Each GPU version is
run once as warm-up and then timed. Repeated `REPEATS` times; the median is used.
""")
code('''
import subprocess, pandas as pd
REPEATS, ITERS = 3, 50
cols = ['file','V','E','iters','cpu_ms','pull_kernel_ms','pull_total_ms',
        'warp_kernel_ms','warp_total_ms','push_kernel_ms','push_total_ms']
rows = []
for pct in [10, 25, 50, 100]:
    for r in range(REPEATS):
        out = subprocess.run(['./pagerank', f'data/graph_{pct}.bin', '-', str(ITERS)],
                             capture_output=True, text=True).stdout
        print(out)
        line = [l for l in out.splitlines() if l.startswith('RESULT,')][0]
        vals = line.split(',')[1:]
        row = dict(zip(cols, vals)); row['pct'] = pct
        rows.append(row)
df = pd.DataFrame(rows)
for c in cols[1:]:
    df[c] = pd.to_numeric(df[c])
res = df.groupby('pct').median(numeric_only=True).reset_index()
for v in ['pull', 'warp', 'push']:
    res[f'{v}_speedup'] = res.cpu_ms / res[f'{v}_kernel_ms']
    res[f'{v}_speedup_total'] = res.cpu_ms / res[f'{v}_total_ms']
gpu = subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
res['gpu'] = gpu
res.to_csv('results/benchmark.csv', index=False)
pd.set_option('display.width', 200)
res
''')

md("## 7. Charts for the slides")
code('''
import matplotlib.pyplot as plt, numpy as np
x = np.arange(len(res)); w = 0.25
labels = [f"{p}%\\n{int(e/1e6*10)/10}M links" for p, e in zip(res.pct, res.E)]

fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (v, name) in enumerate([('pull','pull (thread/page)'), ('warp','warp/page'), ('push','push (atomic)')]):
    b = ax.bar(x + (i-1)*w, res[f'{v}_speedup'], w, label=name)
    ax.bar_label(b, fmt='%.1fx', fontsize=8)
ax.set_xticks(x, labels); ax.set_ylabel('speedup over CPU (kernels only)')
ax.set_title(f'GPU speedup vs graph size ({gpu}, {ITERS} iterations)')
ax.legend(); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('results/speedup.png', dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(res.E/1e6, res.cpu_ms, 'o-', label='CPU')
for v, name in [('pull','pull'), ('warp','warp/page'), ('push','push')]:
    ax.plot(res.E/1e6, res[f'{v}_kernel_ms'], 'o-', label=f'GPU {name}')
ax.set_xlabel('links (millions)'); ax.set_ylabel(f'time for {ITERS} iterations (ms)')
ax.set_yscale('log'); ax.set_title('Time vs graph size'); ax.legend()
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('results/time.png', dpi=200); plt.show()
''')

md("""
## 8. Save results
Download `results/` (or it is copied to Drive if `USE_DRIVE = True`) and send
`benchmark.csv`, `correctness.txt` and the two PNGs back for the slides.
""")
code('''
!ls -lh results
if CACHE:
    !cp -r results {CACHE}/
else:
    from google.colab import files
    !zip -q -r results.zip results
    files.download('results.zip')
''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(os.path.join(HERE, "pagerank_thwiki.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote pagerank_thwiki.ipynb with", len(cells), "cells")
