#define K 10
#define THREADS 128
#define NEG_INF (-3.402823466e+38f)

// Insert (s, j) into a list kept sorted in descending order.
__device__ __forceinline__ void insert(float* v, int* ix, float s, int j) {
    if (s <= v[K - 1]) return;
    int p = K - 1;
    while (p > 0 && v[p - 1] < s) {
        v[p] = v[p - 1];
        ix[p] = ix[p - 1];
        --p;
    }
    v[p] = s;
    ix[p] = j;
}

// One block per row of S: each thread scans a strided slice keeping its own top-K,
// then the per-thread lists are merged pairwise in shared memory (log2(THREADS) rounds).
extern "C" __global__ void topk_rows(const float* __restrict__ S, int n_cols, int* __restrict__ out) {
    __shared__ float sv[THREADS * K];
    __shared__ int si[THREADS * K];
    const int tid = threadIdx.x;
    const float* row = S + (size_t)blockIdx.x * n_cols;

    float v[K];
    int ix[K];
    for (int t = 0; t < K; ++t) { v[t] = NEG_INF; ix[t] = -1; }
    for (int j = tid; j < n_cols; j += THREADS) insert(v, ix, row[j], j);
    for (int t = 0; t < K; ++t) { sv[tid * K + t] = v[t]; si[tid * K + t] = ix[t]; }
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float* av = sv + tid * K;
            const float* bv = sv + (tid + stride) * K;
            const int* ai = si + tid * K;
            const int* bi = si + (tid + stride) * K;
            int p = 0, q = 0;
            for (int t = 0; t < K; ++t) {
                if (av[p] >= bv[q]) { v[t] = av[p]; ix[t] = ai[p]; ++p; }
                else                { v[t] = bv[q]; ix[t] = bi[q]; ++q; }
            }
            for (int t = 0; t < K; ++t) { sv[tid * K + t] = v[t]; si[tid * K + t] = ix[t]; }
        }
        __syncthreads();
    }

    if (tid < K) out[(size_t)blockIdx.x * K + tid] = si[tid];
}
