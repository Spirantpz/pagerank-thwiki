// pagerank.cu - PageRank on the Thai Wikipedia link graph, CPU vs GPU.
//
// Build:  nvcc -O3 -arch=sm_75 -o pagerank pagerank.cu      (sm_75 = Colab T4)
// Run:    ./pagerank data/graph_100.bin [data/titles.txt | -] [fixed_iters]
//         fixed_iters > 0 runs exactly that many iterations on CPU and GPU
//         (same work on both = fair timing); otherwise stop when converged.
//
// Graph file (written by prep/convert.py), all little-endian int32:
//     V, E, row_ptr[V+1], col_idx[E], out_deg[V]
// row_ptr/col_idx is CSR of INCOMING links: the pages that link to v are
// col_idx[row_ptr[v]] .. col_idx[row_ptr[v+1]-1].
//
// PageRank (d = 0.85), one iteration:
//     contrib[u] = rank[u] / out_deg[u]            (0 for dangling pages)
//     D          = sum of rank[u] over dangling u  (their rank is spread over all pages)
//     new[v]     = (1-d)/V + d * ( sum over u->v of contrib[u]  +  D/V )
// Repeat until sum |new - rank| < TOL or MAX_ITER iterations.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif

#define DAMPING   0.85f
#define TOL       1e-5      // float sums of ~10^4 terms cannot go much lower
#define MAX_ITER  100
#define BLOCK_SIZE 256
#define TOP_K     20

#ifndef CPU_ONLY
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err),      \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// Graph loading
// ---------------------------------------------------------------------------
typedef struct {
    int V, E;
    int *row_ptr;   // V+1
    int *col_idx;   // E   source of each incoming link
    int *out_deg;   // V
} Graph;

static Graph load_graph(const char *path)
{
    Graph g;
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    if (fread(&g.V, 4, 1, f) != 1 || fread(&g.E, 4, 1, f) != 1) { printf("bad file\n"); exit(1); }
    g.row_ptr = (int *)malloc((size_t)(g.V + 1) * sizeof(int));
    g.col_idx = (int *)malloc((size_t)g.E * sizeof(int));
    g.out_deg = (int *)malloc((size_t)g.V * sizeof(int));
    if (fread(g.row_ptr, 4, g.V + 1, f) != (size_t)(g.V + 1) ||
        fread(g.col_idx, 4, g.E, f) != (size_t)g.E ||
        fread(g.out_deg, 4, g.V, f) != (size_t)g.V) { printf("truncated file\n"); exit(1); }
    fclose(f);
    return g;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// ---------------------------------------------------------------------------
// Sequential PageRank (the CPU baseline)
// ---------------------------------------------------------------------------
// One iteration reads every vertex and every edge exactly once: O(V + E).
// Any correct algorithm must read every edge at least once per iteration
// (each edge changes a rank), so this is optimal per iteration.
// Pull order (loop over v, read its incoming list) writes each new[v] once,
// sequentially, and precomputing contrib[] replaces E divisions by V.
static int pagerank_cpu(const Graph *g, float *rank, int fixed)
{
    int V = g->V;
    float *contrib = (float *)malloc(V * sizeof(float));
    float *next = (float *)malloc(V * sizeof(float));
    float base = (1.0f - DAMPING) / V;

    for (int v = 0; v < V; v++)
        rank[v] = 1.0f / V;

    int maxit = fixed > 0 ? fixed : MAX_ITER;
    int iter;
    for (iter = 1; iter <= maxit; iter++) {
        // map: contrib, and reduce: dangling mass
        double dangling = 0.0;
        for (int u = 0; u < V; u++) {
            if (g->out_deg[u] > 0)
                contrib[u] = rank[u] / g->out_deg[u];
            else {
                contrib[u] = 0.0f;
                dangling += rank[u];
            }
        }
        float dterm = DAMPING * (float)(dangling / V);

        // pull: gather incoming contributions, and reduce: L1 change
        double diff = 0.0;
        for (int v = 0; v < V; v++) {
            float sum = 0.0f;
            for (int j = g->row_ptr[v]; j < g->row_ptr[v + 1]; j++)
                sum += contrib[g->col_idx[j]];
            next[v] = base + dterm + DAMPING * sum;
            diff += fabsf(next[v] - rank[v]);
        }

        float *t = rank; rank = next; next = t;      // swap pointers, no copy
        if (fixed <= 0 && diff < TOL)
            break;
    }
    if (iter > maxit) iter = maxit;
    // after an odd number of swaps the result is in the buffer we malloc'd
    if (iter % 2 == 1) { memcpy(next, rank, V * sizeof(float)); free(rank); }
    else free(next);
    free(contrib);
    return iter;
}

#ifndef CPU_ONLY
// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Block-level sum in shared memory (reduction from the lectures, K2 style:
// active threads stay contiguous), then ONE atomicAdd per block (privatization).
__device__ void block_sum_atomic(double val, double *out)
{
    __shared__ double s[BLOCK_SIZE];
    s[threadIdx.x] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, s[0]);       // double atomicAdd needs sm_60+
}

// Step 1: map contrib[u] = rank[u]/out_deg[u], and reduce the dangling mass.
__global__ void contribKernel(const float *rank, const int *out_deg, float *contrib,
                              double *dangling, int V)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    double d = 0.0;
    if (u < V) {
        int k = out_deg[u];
        if (k > 0) contrib[u] = rank[u] / k;
        else { contrib[u] = 0.0f; d = rank[u]; }
    }
    block_sum_atomic(d, dangling);
}

// Step 2 (version A, "pull"): one thread per page v, sums its incoming links.
// Each thread writes only next[v], so there are no races and no atomics.
__global__ void pullKernel(const int *row_ptr, const int *col_idx, const float *contrib,
                           const float *rank, float *next, const double *dangling,
                           double *diff, int V)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double change = 0.0;
    if (v < V) {
        float sum = 0.0f;
        for (int j = row_ptr[v]; j < row_ptr[v + 1]; j++)
            sum += contrib[col_idx[j]];
        float base = (1.0f - DAMPING) / V + DAMPING * (float)(*dangling / V);
        float r = base + DAMPING * sum;
        next[v] = r;
        change = fabsf(r - rank[v]);
    }
    block_sum_atomic(change, diff);
}

// Step 2 (version B, "warp per page"): 32 threads share one page's incoming list,
// so a page with 10,000 in-links is not left to a single slow thread.
// Lanes read neighbouring col_idx entries -> coalesced; then a warp shuffle sum.
__global__ void warpKernel(const int *row_ptr, const int *col_idx, const float *contrib,
                           const float *rank, float *next, const double *dangling,
                           double *diff, int V)
{
    int lane = threadIdx.x & 31;
    int v = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;    // one warp = one page
    double change = 0.0;
    if (v < V) {
        float sum = 0.0f;
        for (int j = row_ptr[v] + lane; j < row_ptr[v + 1]; j += 32)
            sum += contrib[col_idx[j]];
        for (int off = 16; off > 0; off >>= 1)                // tree inside the warp
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) {
            float base = (1.0f - DAMPING) / V + DAMPING * (float)(*dangling / V);
            float r = base + DAMPING * sum;
            next[v] = r;
            change = fabsf(r - rank[v]);
        }
    }
    block_sum_atomic(change, diff);
}

// Step 2 (version C, "push"): one thread per LINK, adds its contribution to the
// target with atomicAdd. Perfect load balance, but popular pages are hit by many
// threads at once -> atomic contention (the histogram problem from the lectures).
__global__ void pushKernel(const int *edge_dst, const int *col_idx, const float *contrib,
                           float *sum, int E)
{
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < (size_t)E; j += stride)
        atomicAdd(&sum[edge_dst[j]], contrib[col_idx[j]]);
}

// Finish for push: new rank from the summed contributions.
__global__ void applyKernel(const float *sum, const float *rank, float *next,
                            const double *dangling, double *diff, int V)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double change = 0.0;
    if (v < V) {
        float base = (1.0f - DAMPING) / V + DAMPING * (float)(*dangling / V);
        float r = base + DAMPING * sum[v];
        next[v] = r;
        change = fabsf(r - rank[v]);
    }
    block_sum_atomic(change, diff);
}

__global__ void initKernel(float *rank, int V)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < V) rank[v] = 1.0f / V;
}

// ---------------------------------------------------------------------------
// GPU driver: the graph is copied once and stays on the GPU for all iterations.
// Only 8 bytes (the L1 change) come back per iteration, for the stop test.
// ---------------------------------------------------------------------------
enum { PULL = 0, WARP = 1, PUSH = 2 };
static const char *NAMES[] = { "pull (thread/page)", "warp/page", "push (atomic)" };

static int pagerank_gpu(const Graph *g, float *h_rank, int version, int fixed,
                        float *kernel_ms, float *total_ms)
{
    int V = g->V, E = g->E;
    cudaEvent_t t0, t1, k0, k1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventCreate(&k0)); CHECK(cudaEventCreate(&k1));
    CHECK(cudaEventRecord(t0));

    int *d_row, *d_col, *d_deg, *d_dst = NULL;
    float *d_rank, *d_next, *d_contrib, *d_sum = NULL;
    double *d_dangling, *d_diff;
    CHECK(cudaMalloc(&d_row, (size_t)(V + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_col, (size_t)E * sizeof(int)));
    CHECK(cudaMalloc(&d_deg, (size_t)V * sizeof(int)));
    CHECK(cudaMalloc(&d_rank, (size_t)V * sizeof(float)));
    CHECK(cudaMalloc(&d_next, (size_t)V * sizeof(float)));
    CHECK(cudaMalloc(&d_contrib, (size_t)V * sizeof(float)));
    CHECK(cudaMalloc(&d_dangling, sizeof(double)));
    CHECK(cudaMalloc(&d_diff, sizeof(double)));

    CHECK(cudaMemcpy(d_row, g->row_ptr, (size_t)(V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col, g->col_idx, (size_t)E * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_deg, g->out_deg, (size_t)V * sizeof(int), cudaMemcpyHostToDevice));

    if (version == PUSH) {
        // push needs the target of every link: expand row_ptr into one entry per link
        int *h_dst = (int *)malloc((size_t)E * sizeof(int));
        for (int v = 0; v < V; v++)
            for (int j = g->row_ptr[v]; j < g->row_ptr[v + 1]; j++)
                h_dst[j] = v;
        CHECK(cudaMalloc(&d_dst, (size_t)E * sizeof(int)));
        CHECK(cudaMalloc(&d_sum, (size_t)V * sizeof(float)));
        CHECK(cudaMemcpy(d_dst, h_dst, (size_t)E * sizeof(int), cudaMemcpyHostToDevice));
        free(h_dst);
    }

    int grid = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int warpGrid = (int)(((size_t)V * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int edgeGrid = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (edgeGrid > 1024) edgeGrid = 1024;
    if (edgeGrid < 1) edgeGrid = 1;

    CHECK(cudaEventRecord(k0));
    initKernel<<<grid, BLOCK_SIZE>>>(d_rank, V);

    int maxit = fixed > 0 ? fixed : MAX_ITER;
    int iter;
    double h_diff = 0.0;
    for (iter = 1; iter <= maxit; iter++) {
        CHECK(cudaMemset(d_dangling, 0, sizeof(double)));
        CHECK(cudaMemset(d_diff, 0, sizeof(double)));
        contribKernel<<<grid, BLOCK_SIZE>>>(d_rank, d_deg, d_contrib, d_dangling, V);

        if (version == PULL)
            pullKernel<<<grid, BLOCK_SIZE>>>(d_row, d_col, d_contrib, d_rank, d_next,
                                             d_dangling, d_diff, V);
        else if (version == WARP)
            warpKernel<<<warpGrid, BLOCK_SIZE>>>(d_row, d_col, d_contrib, d_rank, d_next,
                                                 d_dangling, d_diff, V);
        else {
            CHECK(cudaMemset(d_sum, 0, (size_t)V * sizeof(float)));
            pushKernel<<<edgeGrid, BLOCK_SIZE>>>(d_dst, d_col, d_contrib, d_sum, E);
            applyKernel<<<grid, BLOCK_SIZE>>>(d_sum, d_rank, d_next, d_dangling, d_diff, V);
        }
        CHECK(cudaGetLastError());

        // 8-byte copy back; this also waits for the kernels to finish
        CHECK(cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost));
        float *t = d_rank; d_rank = d_next; d_next = t;
        if (fixed <= 0 && h_diff < TOL)
            break;
    }
    if (iter > maxit) iter = maxit;
    CHECK(cudaEventRecord(k1));

    CHECK(cudaMemcpy(h_rank, d_rank, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_row)); CHECK(cudaFree(d_col)); CHECK(cudaFree(d_deg));
    CHECK(cudaFree(d_rank)); CHECK(cudaFree(d_next)); CHECK(cudaFree(d_contrib));
    CHECK(cudaFree(d_dangling)); CHECK(cudaFree(d_diff));
    if (d_dst) { CHECK(cudaFree(d_dst)); CHECK(cudaFree(d_sum)); }
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));

    CHECK(cudaEventElapsedTime(kernel_ms, k0, k1));   // iterations only
    CHECK(cudaEventElapsedTime(total_ms, t0, t1));    // + malloc, copies, free
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(k0); cudaEventDestroy(k1);
    return iter;
}
#endif

// ---------------------------------------------------------------------------
// Checking and printing
// ---------------------------------------------------------------------------
static void top_k(const float *rank, int V, int *idx, int k)
{
    // simple selection: k passes over V, fine for k = 20
    for (int i = 0; i < k; i++) {
        int best = -1;
        for (int v = 0; v < V; v++) {
            int taken = 0;
            for (int j = 0; j < i; j++) if (idx[j] == v) { taken = 1; break; }
            if (!taken && (best < 0 || rank[v] > rank[best])) best = v;
        }
        idx[i] = best;
    }
}

static char **load_titles(const char *path, int V)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    char **t = (char **)calloc(V, sizeof(char *));
    char line[1024];
    for (int i = 0; i < V && fgets(line, sizeof line, f); i++) {
        line[strcspn(line, "\n")] = 0;
        t[i] = strdup(line);
    }
    fclose(f);
    return t;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s graph.bin [titles.txt | -] [fixed_iters]\n", argv[0]);
        return 1;
    }
    Graph g = load_graph(argv[1]);
    int fixed = argc >= 4 ? atoi(argv[3]) : 0;
    int V = g.V, k = V < TOP_K ? V : TOP_K;
    int dangling = 0, maxin = 0;
    for (int v = 0; v < V; v++) {
        if (g.out_deg[v] == 0) dangling++;
        int in = g.row_ptr[v + 1] - g.row_ptr[v];
        if (in > maxin) maxin = in;
    }
    printf("graph %s: V=%d E=%d dangling=%d max in-degree=%d\n",
           argv[1], V, g.E, dangling, maxin);

    // CPU
    float *cpu = (float *)malloc(V * sizeof(float));
    double c0 = now_ms();
    int cpu_iter = pagerank_cpu(&g, cpu, fixed);
    double cpu_ms = now_ms() - c0;
    printf("CPU                 : %3d iterations  %10.2f ms\n", cpu_iter, cpu_ms);

    int cpu_top[TOP_K];
    top_k(cpu, V, cpu_top, k);

#ifndef CPU_ONLY
    float *gpu = (float *)malloc(V * sizeof(float));
    float kms[3], tms[3];
    int ok_all = 1;
    for (int ver = 0; ver < 3; ver++) {
        float kernel_ms, total_ms;
        pagerank_gpu(&g, gpu, ver, fixed, &kernel_ms, &total_ms);   // warm-up (CUDA context, caches)
        int it = pagerank_gpu(&g, gpu, ver, fixed, &kernel_ms, &total_ms);
        kms[ver] = kernel_ms; tms[ver] = total_ms;

        // correctness: largest RELATIVE per-page difference, and the same top 20.
        // Float sums depend on the order of additions, so CPU and GPU cannot be
        // bit-identical; 1e-3 relative is far below what changes the ranking.
        double maxerr = 0.0;
        for (int v = 0; v < V; v++) {
            double e = fabs((double)gpu[v] - cpu[v]) / cpu[v];
            if (e > maxerr) maxerr = e;
        }
        int gpu_top[TOP_K];
        top_k(gpu, V, gpu_top, k);
        // same top 20, in the same order; two pages may swap only if their CPU
        // ranks are a tie (differ by < 0.01%), because float order can flip a tie
        int same = 1, ties = 0;
        for (int i = 0; i < k; i++) {
            if (gpu_top[i] == cpu_top[i]) continue;
            double a = cpu[gpu_top[i]], b = cpu[cpu_top[i]];
            if (fabs(a - b) / b < 1e-4) ties++;
            else same = 0;
        }
        int ok = maxerr < 1e-3 && same;
        ok_all &= ok;
        printf("GPU %-16s: %3d iterations  %10.2f ms kernels  %10.2f ms total  "
               "speedup %6.2fx (%5.2fx incl. copies)  max rel err %.1e  top%d %s\n",
               NAMES[ver], it, kernel_ms, total_ms, cpu_ms / kernel_ms, cpu_ms / total_ms,
               maxerr, k, !same ? "DIFFERENT" : ties ? "same (tie swapped)" : "same");
    }
    printf("check: %s\n", ok_all ? "PASS" : "FAIL");
    // one machine-readable line for the notebook
    printf("RESULT,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", argv[1], V, g.E, cpu_iter, cpu_ms,
           kms[0], tms[0], kms[1], tms[1], kms[2], tms[2]);
    free(gpu);
#endif

    if (argc >= 3 && strcmp(argv[2], "-") != 0) {
        char **titles = load_titles(argv[2], V);
        if (titles) {
            printf("\nTop %d pages (CPU ranks):\n", k);
            for (int i = 0; i < k; i++)
                printf("%2d. %.6f  %s\n", i + 1, cpu[cpu_top[i]],
                       titles[cpu_top[i]] ? titles[cpu_top[i]] : "?");
        }
    }
    free(cpu); free(g.row_ptr); free(g.col_idx); free(g.out_deg);
    return 0;
}
