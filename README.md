# PageRank on the Thai Wikipedia link graph — CPU vs GPU (CUDA)

Term project, Parallel Computing. Computes PageRank for every article of Thai
Wikipedia (188,430 articles, 11.1 million links) from its link graph, sequentially on the CPU and in
parallel on the GPU with CUDA, and compares speed and results.

## Files

| File | What it is |
| --- | --- |
| `prep/convert.py` | Turns the Wikipedia SQL dump into a graph file (CSR of incoming links) |
| `pagerank.cu` | Sequential CPU PageRank + three GPU versions (pull, warp-per-page, push) + timing and correctness check |
| `pagerank_thwiki.ipynb` | Google Colab notebook that runs everything end to end |
| `make_notebook.py` | Rebuilds the notebook from the two source files above |
| `results/` | Benchmark CSV and correctness output from the Colab run |
| `slides/` | Presentation (`pagerank_thwiki.pptx` / `.pdf`), Thai speaking script, Q&A practice |

## How to run

### Easiest: Google Colab
1. Open `pagerank_thwiki.ipynb` in Colab (File → Upload notebook).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**. It downloads the dump (about 180 MB), builds the
   graph (about 5 minutes), compiles, checks correctness, benchmarks, and draws the charts.

### Any Linux machine with an NVIDIA GPU (compute capability 6.0+)
```bash
# 1. data
mkdir -p dump data
for t in page redirect pagelinks linktarget; do
  wget -P dump https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-$t.sql.gz
done
pip install numpy
python3 prep/convert.py --dump dump --out data

# 2. build (change sm_75 to your GPU: sm_61 GTX 10xx, sm_75 T4/RTX 20xx, sm_86 RTX 30xx, sm_89 RTX 40xx)
nvcc -O3 -arch=sm_75 -o pagerank pagerank.cu

# 3. run until converged, print top-20 pages
./pagerank data/graph_100.bin data/titles.txt

# 4. benchmark: exactly 50 iterations on CPU and GPU (same work)
for p in 10 25 50 100; do ./pagerank data/graph_$p.bin - 50; done
```

`./pagerank <graph.bin> [titles.txt | -] [fixed_iters]`

## Dataset

Thai Wikipedia database dump, https://dumps.wikimedia.org/thwiki/latest/
(tables `page`, `redirect`, `pagelinks`, `linktarget`). Only articles
(namespace 0) are kept; redirects are resolved to their target article;
duplicate links and self-links are removed. `graph_50/25/10.bin` are induced
subgraphs on a random 50/25/10 % of the articles (seed 42), used for the
speedup-vs-size chart.

## Algorithm

PageRank with damping d = 0.85, power iteration:

new[v] = (1 − d)/V + d · ( Σ<sub>u→v</sub> rank[u]/outdeg[u] + D/V ),
where D is the total rank of pages with no outgoing links.
Stops when Σ|new − rank| < 1e-5 (or after a fixed number of iterations for benchmarking).

GPU, per iteration, with the graph kept on the GPU the whole time:
1. `contribKernel` — map rank/outdeg, plus a block reduction of the dangling mass
2. one of
   - `pullKernel` — one thread per page sums its incoming links (no atomics)
   - `warpKernel` — one warp per page, lanes split the incoming list, shuffle reduction
   - `pushKernel` + `applyKernel` — one thread per link, `atomicAdd` into the target
3. block reduction of the L1 change; 8 bytes copied back for the stop test

## Results (Google Colab, Tesla T4, CUDA 13.0, dump of 1 Oct 2026)

Full graph: V = 188,430, E = 11,110,180, 338 dangling pages, max in-degree 59,291
(ภาษาอังกฤษ). Converges in 32 iterations; all three GPU versions match the CPU
(max relative error ≤ 1.4e-5, same top 20) → `check: PASS`.

Time for exactly 50 iterations (median of 3 runs), kernels only, in ms:

| Graph | V | E | CPU | A pull | B warp/page | C push | best speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10%  | 18,843  | 106,334    | 16.6  | 4.6   | 10.0 | 2.5  | 6.6× (C) |
| 25%  | 47,107  | 679,127    | 60.8  | 12.8  | 21.0 | 8.3  | 7.3× (C) |
| 50%  | 94,215  | 2,763,248  | 201.5 | 23.4  | 29.0 | 20.7 | 9.7× (C) |
| 100% | 188,430 | 11,110,180 | 822.1 | 127.8 | 59.1 | 61.0 | 13.9× (B) |

Including all host↔device copies, B is 11.6× faster than the CPU on the full graph.
Raw numbers: `results/benchmark.csv`, `results/correctness.txt`.

Top 5 articles: ภาษาอังกฤษ, เวย์แบ็กแมชชีน, เลขมาตรฐานสากลประจำหนังสือ,
ตัวระบุวัตถุดิจิทัล, สหรัฐอเมริกา.
