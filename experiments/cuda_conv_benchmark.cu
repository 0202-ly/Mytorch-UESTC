#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void conv2d_nchw(const float* x, const float* w, const float* b,
                            float* y, int N, int C, int H, int W, int K,
                            int R, int S, int OH, int OW) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N * K * OH * OW;
    if (idx >= total) return;
    int ow = idx % OW; idx /= OW;
    int oh = idx % OH; idx /= OH;
    int k = idx % K; idx /= K;
    int n = idx;
    float acc = b[k];
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s) {
                int xi = ((n * C + c) * H + oh + r) * W + ow + s;
                int wi = ((k * C + c) * R + r) * S + s;
                acc += x[xi] * w[wi];
            }
    y[((n * K + k) * OH + oh) * OW + ow] = acc;
}

static void check(cudaError_t code, const char* where) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(code));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    int repeats = argc > 1 ? std::atoi(argv[1]) : 50;
    struct Case { int n, c, h, w, k, r, s; } cases[] = {
        {1, 3, 16, 16, 8, 3, 3},
        {4, 3, 32, 32, 16, 3, 3},
        {8, 3, 64, 64, 32, 3, 3},
    };
    std::printf("backend,case,n,c,h,w,k,r,s,mean_ms,max_abs_error,relative_error\n");
    for (int ci = 0; ci < 3; ++ci) {
        auto q = cases[ci];
        int oh = q.h - q.r + 1, ow = q.w - q.s + 1;
        size_t nx = (size_t)q.n * q.c * q.h * q.w;
        size_t nw = (size_t)q.k * q.c * q.r * q.s;
        size_t ny = (size_t)q.n * q.k * oh * ow;
        std::vector<float> hx(nx, 0.01f), hw(nw, 0.02f), hb(q.k, 0.03f);
        float *dx, *dw, *db, *dy;
        check(cudaMalloc(&dx, nx*sizeof(float)), "cudaMalloc x");
        check(cudaMalloc(&dw, nw*sizeof(float)), "cudaMalloc w");
        check(cudaMalloc(&db, q.k*sizeof(float)), "cudaMalloc b");
        check(cudaMalloc(&dy, ny*sizeof(float)), "cudaMalloc y");
        check(cudaMemcpy(dx, hx.data(), nx*sizeof(float), cudaMemcpyHostToDevice), "copy x");
        check(cudaMemcpy(dw, hw.data(), nw*sizeof(float), cudaMemcpyHostToDevice), "copy w");
        check(cudaMemcpy(db, hb.data(), q.k*sizeof(float), cudaMemcpyHostToDevice), "copy b");
        int threads = 256, blocks = (int)((ny + threads - 1) / threads);
        for (int i = 0; i < 10; ++i) conv2d_nchw<<<blocks, threads>>>(dx,dw,db,dy,q.n,q.c,q.h,q.w,q.k,q.r,q.s,oh,ow);
        check(cudaDeviceSynchronize(), "warmup");
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < repeats; ++i) conv2d_nchw<<<blocks, threads>>>(dx,dw,db,dy,q.n,q.c,q.h,q.w,q.k,q.r,q.s,oh,ow);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop); ms /= repeats;
        std::vector<float> hy(ny);
        check(cudaMemcpy(hy.data(), dy, ny*sizeof(float), cudaMemcpyDeviceToHost), "copy y");
        double max_abs = 0.0, max_rel = 0.0;
        for (int n = 0; n < q.n; ++n) for (int k = 0; k < q.k; ++k)
            for (int oh_i = 0; oh_i < oh; ++oh_i) for (int ow_i = 0; ow_i < ow; ++ow_i) {
                double ref = hb[k];
                for (int c = 0; c < q.c; ++c) for (int r_i = 0; r_i < q.r; ++r_i) for (int s_i = 0; s_i < q.s; ++s_i) {
                    int xi = ((n*q.c+c)*q.h+oh_i+r_i)*q.w+ow_i+s_i;
                    int wi = ((k*q.c+c)*q.r+r_i)*q.s+s_i;
                    ref += static_cast<double>(hx[xi]) * static_cast<double>(hw[wi]);
                }
                double diff = std::fabs(static_cast<double>(hy[((n*q.k+k)*oh+oh_i)*ow+ow_i]) - ref);
                max_abs = std::max(max_abs, diff);
                max_rel = std::max(max_rel, diff / std::max(std::fabs(ref), 1e-6));
            }
        std::printf("cuda_c,%d,%d,%d,%d,%d,%d,%d,%d,%.8f,%.8g,%.8g\n",ci,q.n,q.c,q.h,q.w,q.k,q.r,q.s,ms,max_abs,max_rel);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dy);
    }
    return 0;
}
