/*********************************************************************
 * selectGoodFeatures_cuda.cu — streamlined + actually faster
 *
 * - Eigenvalue window sums on GPU (deterministic indexing; no atomics)
 * - In-place device bitonic sort by val desc (with proper tail padding)
 * - Copy sorted list once, enforce mindist on CPU (simple & robust)
 *********************************************************************/

#include <assert.h>
 #include <stdlib.h>
 #include <stdio.h>
 #include <string.h>
 #include <math.h>
 #include <limits.h>
#include <cuda_runtime.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"

 int KLT_verbose = 0;
 typedef enum { SELECTING_ALL, REPLACING_SOME } selectionMode;
 
 #ifdef ENABLE_NVTX
   #include <nvToolsExt.h>
   #define NVTX_PUSH(n) nvtxRangePushA(n)
   #define NVTX_POP()   nvtxRangePop()
 #else
   #define NVTX_PUSH(n) ((void)0)
   #define NVTX_POP()   ((void)0)
 #endif
 
 #define CUDA_CHECK(call) \
   do { cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
          fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
          exit(EXIT_FAILURE); } \
   } while(0)
 
 // ------------------------------------------------------------------
 // Persistent device state
 // ------------------------------------------------------------------
 static struct {
   int*          d_pointlist;    // triplets [x,y,val], length = max_points
   size_t        max_points;
   cudaStream_t  stream;
   int           init;
 } g_sgf = {nullptr, 0, 0, 0};
 
 static void sgf_ensure(size_t need_points) {
   if (!g_sgf.init) {
     CUDA_CHECK(cudaStreamCreate(&g_sgf.stream));
     g_sgf.init = 1;
   }
   if (need_points > g_sgf.max_points) {
     if (g_sgf.d_pointlist) cudaFree(g_sgf.d_pointlist);
     CUDA_CHECK(cudaMalloc(&g_sgf.d_pointlist, need_points * 3 * sizeof(int)));
     g_sgf.max_points = need_points;
   }
 }
 
 // ------------------------------------------------------------------
 // Device helpers
 // ------------------------------------------------------------------
 static __device__ __forceinline__ float rd(const float* p) {
 #if __CUDA_ARCH__ >= 350
   return __ldg(p);
 #else
   return *p;
 #endif
 }
 
 // Each thread computes one candidate (xi,yi) -> (x,y) and the min-eigenvalue
 __global__ void eigen_kernel_grid(
   const float* __restrict__ gradx,
   const float* __restrict__ grady,
   int ncols, int nrows,
   int window_hw, int window_hh,
   int borderx, int bordery,
   int skip,                     // stride in candidate grid: (skip+1)
   int x_count, int y_count,     // number of candidate points along x/y
   int* __restrict__ out_triplets) // [x,y,val]
 {
   int xi = blockIdx.x * blockDim.x + threadIdx.x;
   int yi = blockIdx.y * blockDim.y + threadIdx.y;
   if (xi >= x_count || yi >= y_count) return;
 
   const int stride = skip + 1;
 
   const int x = borderx + xi * stride;
   const int y = bordery + yi * stride;
 
   float gxx = 0.f, gxy = 0.f, gyy = 0.f;
 
 #pragma unroll 2
   for (int yy = y - window_hh; yy <= y + window_hh; ++yy) {
     const int base = yy * ncols;
 #pragma unroll 2
     for (int xx = x - window_hw; xx <= x + window_hw; ++xx) {
       const int idx = base + xx;
       const float gx = rd(gradx + idx);
       const float gy = rd(grady + idx);
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
            }
        }
        
   const float trace = gxx + gyy;
   const float det   = gxx * gyy - gxy * gxy;
   const float disc  = fmaxf(trace * trace - 4.f * det, 0.f);
   const float minev = 0.5f * (trace - sqrtf(disc));
 
   const int val = (int)fminf(minev, (float)(INT_MAX - 1));
 
   const int out_idx = yi * x_count + xi;
   const int t = out_idx * 3;
   out_triplets[t + 0] = x;
   out_triplets[t + 1] = y;
   out_triplets[t + 2] = val;
 }
 
 // Simple, fast in-place bitonic sort on device triplets by val desc
 __global__ void bitonic_step_kernel(int* data, int n, unsigned j, unsigned k)
 {
   unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= (unsigned)n) return;
 
   unsigned ixj = i ^ j;
   if (ixj <= i || ixj >= (unsigned)n) return;
 
   const int vi = data[3 * i + 2];
   const int vj = data[3 * ixj + 2];
 
   const bool ascending_half = ((i & k) == 0);
   const bool swap = ascending_half ? (vi < vj) : (vi > vj);
   if (swap) {
     int tx = data[3 * i + 0], ty = data[3 * i + 1], tv = data[3 * i + 2];
     data[3 * i + 0]    = data[3 * ixj + 0];
     data[3 * i + 1]    = data[3 * ixj + 1];
     data[3 * i + 2]    = data[3 * ixj + 2];
     data[3 * ixj + 0]  = tx;
     data[3 * ixj + 1]  = ty;
     data[3 * ixj + 2]  = tv;
   }
 }
 
 static inline unsigned next_pow2(unsigned x) {
   if (x <= 1u) return 1u;
   --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
   return x + 1;
 }
 
 static void device_bitonic_sort_triplets_desc(int* d_triplets, int n, cudaStream_t s)
 {
   if (n <= 1) return;
   const unsigned m = next_pow2((unsigned)n);
   const int block = 256;
   const int grid  = (int)((m + block - 1) / block);
 
   for (unsigned k = 2; k <= m; k <<= 1) {
     for (unsigned j = k >> 1; j > 0; j >>= 1) {
       bitonic_step_kernel<<<grid, block, 0, s>>>(d_triplets, (int)m, j, k);
       CUDA_CHECK(cudaGetLastError());
     }
   }
   CUDA_CHECK(cudaStreamSynchronize(s)); // catch OOB immediately
 }
 
 // Pad tail triplets [start .. start+count) with sentinel {0,0,INT_MIN}
 __global__ void pad_tail_kernel(int* data, int start, int count)
 {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= count) return;
   const int t = (start + i) * 3;
   data[t + 0] = 0;
   data[t + 1] = 0;
   data[t + 2] = INT_MIN;
 }
 
 // ------------------------------------------------------------------
 // CPU helpers (unchanged logic; simple and fast for N ~ O(1e5))
 // ------------------------------------------------------------------
 static void fill_featuremap_block(int cx, int cy, uchar* fmap, int mindist, int W, int H)
 {
   for (int dy = -mindist; dy <= mindist; ++dy) {
     const int y = cy + dy; if (y < 0 || y >= H) continue;
     const int row = y * W;
     for (int dx = -mindist; dx <= mindist; ++dx) {
       const int x = cx + dx; if (x < 0 || x >= W) continue;
       fmap[row + x] = 1;
     }
   }
 }
 
 static void enforce_min_dist_cpu(
   const int* pointlist, int npoints,
   KLT_FeatureList fl, int ncols, int nrows,
   int mindist, int min_eigenvalue,
   KLT_BOOL overwriteAll)
 {
   if (mindist < 0) mindist = 0;
   uchar* fmap = (uchar*)calloc((size_t)ncols * nrows, sizeof(uchar));
 
   if (!overwriteAll) {
     for (int i = 0; i < fl->nFeatures; ++i) {
       if (fl->feature[i]->val >= 0) {
         fill_featuremap_block((int)fl->feature[i]->x, (int)fl->feature[i]->y,
                               fmap, mindist, ncols, nrows);
       }
     }
   }
 
   int out_idx = 0;
   for (int i = 0; i < npoints; ++i) {
     while (!overwriteAll && out_idx < fl->nFeatures && fl->feature[out_idx]->val >= 0)
       ++out_idx;
     if (out_idx >= fl->nFeatures) break;
 
     const int x   = pointlist[3 * i + 0];
     const int y   = pointlist[3 * i + 1];
     const int val = pointlist[3 * i + 2];
     if (val < min_eigenvalue) continue;
     if (fmap[y * ncols + x])  continue;
 
     KLT_Feature f = fl->feature[out_idx];
     f->x   = (KLT_locType)x;
     f->y   = (KLT_locType)y;
     f->val = val;
     f->aff_img = NULL; f->aff_img_gradx = NULL; f->aff_img_grady = NULL;
     f->aff_x = -1.0f; f->aff_y = -1.0f;
     f->aff_Axx = 1.0f; f->aff_Ayx = 0.0f; f->aff_Axy = 0.0f; f->aff_Ayy = 1.0f;
 
     fill_featuremap_block(x, y, fmap, mindist, ncols, nrows);
     ++out_idx;
   }
 
   while (out_idx < fl->nFeatures) {
     if (overwriteAll || fl->feature[out_idx]->val < 0) {
       KLT_Feature f = fl->feature[out_idx];
       f->x = -1; f->y = -1; f->val = KLT_NOT_FOUND;
       f->aff_img = NULL; f->aff_img_gradx = NULL; f->aff_img_grady = NULL;
       f->aff_x = -1.0f; f->aff_y = -1.0f;
       f->aff_Axx = 1.0f; f->aff_Ayx = 0.0f; f->aff_Axy = 0.0f; f->aff_Ayy = 1.0f;
     }
     ++out_idx;
   }
 
   free(fmap);
 }
 
 // ------------------------------------------------------------------
 // Main GPU-accelerated selection (fast path)
 // ------------------------------------------------------------------
 static void _KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
   KLT_PixelType* img,
   int ncols, int nrows,
   KLT_FeatureList fl,
  selectionMode mode)
{
   NVTX_PUSH("SelectGoodFeatures");
 
   if (tc->window_width  < 3) tc->window_width  = 3;
   if (tc->window_height < 3) tc->window_height = 3;
   if ((tc->window_width  & 1) == 0) ++tc->window_width;
   if ((tc->window_height & 1) == 0) ++tc->window_height;
   const int window_hw = tc->window_width  / 2;
   const int window_hh = tc->window_height / 2;
 
   int borderx = tc->borderx; if (borderx < window_hw) borderx = window_hw;
   int bordery = tc->bordery; if (bordery < window_hh) bordery = window_hh;
 
  _KLT_FloatImage floatimg, gradx, grady;
  KLT_BOOL floatimages_created = FALSE;

   NVTX_PUSH("PrepareImages");
   if (mode == REPLACING_SOME && tc->sequentialMode && tc->pyramid_last) {
     floatimg = ((_KLT_Pyramid)tc->pyramid_last)->img[0];
     gradx    = ((_KLT_Pyramid)tc->pyramid_last_gradx)->img[0];
     grady    = ((_KLT_Pyramid)tc->pyramid_last_grady)->img[0];
   } else {
    floatimages_created = TRUE;
    floatimg = _KLTCreateFloatImage(ncols, nrows);
    gradx    = _KLTCreateFloatImage(ncols, nrows);
    grady    = _KLTCreateFloatImage(ncols, nrows);
 
     if (tc->smoothBeforeSelecting) {
       _KLT_FloatImage tmp = _KLTCreateFloatImage(ncols, nrows);
       _KLTToFloatImage(img, ncols, nrows, tmp);
       _KLTComputeSmoothedImage(tmp, _KLTComputeSmoothSigma(tc), floatimg);
       _KLTFreeFloatImage(tmp);
     } else {
       _KLTToFloatImage(img, ncols, nrows, floatimg);
     }
    // OLD: _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
    // Use GPU-only version - NO D2H transfer!
  }
  NVTX_POP(); // PrepareImages

  const int stride   = tc->nSkippedPixels + 1;
  const int x_range  = ncols - 2 * borderx;
  const int y_range  = nrows - 2 * bordery;

  // CPU-equivalent loop count: for(x=borderx; x < ncols-borderx; x += stride)
  const int x_count  = (x_range > 0) ? ((x_range - 1) / stride + 1) : 0;
  const int y_count  = (y_range > 0) ? ((y_range - 1) / stride + 1) : 0;
  if (x_count == 0 || y_count == 0) {
    if (floatimages_created) {
      _KLTFreeFloatImage(floatimg); _KLTFreeFloatImage(gradx); _KLTFreeFloatImage(grady);
    }
    NVTX_POP();
    return;
  }
  const int npoints = x_count * y_count;
  const unsigned m  = next_pow2((unsigned)npoints);  // <-- for sort capacity

  // Ensure device buffer is big enough for m entries (not just npoints!)
  sgf_ensure((size_t)m);

  NVTX_PUSH("ComputeGradientsGPU");
  // ZERO-COPY: Compute gradients and keep on GPU!
  float *d_gx = nullptr, *d_gy = nullptr;
  _KLTComputeGradientsGPU(floatimg, tc->grad_sigma, ncols, nrows, &d_gx, &d_gy);
  NVTX_POP();
 
   NVTX_PUSH("EigenKernel");
   dim3 block(16,16);
   dim3 grid((x_count + block.x - 1) / block.x,
             (y_count + block.y - 1) / block.y);
   eigen_kernel_grid<<<grid, block, 0, g_sgf.stream>>>(
       d_gx, d_gy,
       ncols, nrows,
       window_hw, window_hh,
       borderx, bordery,
       tc->nSkippedPixels,
       x_count, y_count,
       g_sgf.d_pointlist);
   CUDA_CHECK(cudaGetLastError());
   CUDA_CHECK(cudaStreamSynchronize(g_sgf.stream));
   NVTX_POP();
 
   NVTX_PUSH("DeviceSort");
   if (m > (unsigned)npoints) {
     const int padN   = (int)(m - (unsigned)npoints);
     const int start  = npoints;
     const int bPad   = 256;
     const int gPad   = (padN + bPad - 1) / bPad;
     pad_tail_kernel<<<gPad, bPad, 0, g_sgf.stream>>>(g_sgf.d_pointlist, start, padN);
     CUDA_CHECK(cudaGetLastError());
   }
   device_bitonic_sort_triplets_desc(g_sgf.d_pointlist, (int)m, g_sgf.stream);
   NVTX_POP();
 
   NVTX_PUSH("D2H");
   int* h_triplets = (int*)malloc((size_t)npoints * 3 * sizeof(int));
   CUDA_CHECK(cudaMemcpyAsync(h_triplets, g_sgf.d_pointlist,
                              (size_t)npoints * 3 * sizeof(int),
                              cudaMemcpyDeviceToHost, g_sgf.stream));
   CUDA_CHECK(cudaStreamSynchronize(g_sgf.stream));
   NVTX_POP();
 
  NVTX_PUSH("MinDistCPU");
  enforce_min_dist_cpu(h_triplets, npoints, fl, ncols, nrows,
                       tc->mindist, tc->min_eigenvalue,
                       (mode == SELECTING_ALL));
  NVTX_POP();

  free(h_triplets);
  // NO cudaFree for d_gx, d_gy - they're managed by convolve.cu's g_gpu buffers!

  if (floatimages_created) {
    _KLTFreeFloatImage(floatimg);
    _KLTFreeFloatImage(gradx);
    _KLTFreeFloatImage(grady);
  }
 
   NVTX_POP(); // SelectGoodFeatures
 }
 
 // ------------------------------------------------------------------
 // Public API
 // ------------------------------------------------------------------
void KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
   KLT_PixelType* img,
   int ncols, int nrows,
  KLT_FeatureList fl)
{
   if (KLT_verbose >= 1) {
     fprintf(stderr, "(KLT) Selecting the %d best features from a %d by %d image...  ",
             fl->nFeatures, ncols, nrows);
    fflush(stderr);
  }

   _KLTSelectGoodFeatures(tc, img, ncols, nrows, fl, SELECTING_ALL);

   if (KLT_verbose >= 1) {
     fprintf(stderr, "\n\t%d features found.\n", KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
       fprintf(stderr, "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

void KLTReplaceLostFeatures(
  KLT_TrackingContext tc,
   KLT_PixelType* img,
   int ncols, int nrows,
  KLT_FeatureList fl)
{
   const int nLost = fl->nFeatures - KLTCountRemainingFeatures(fl);
 
   if (KLT_verbose >= 1) {
     fprintf(stderr, "(KLT) Attempting to replace %d features in a %d by %d image...  ",
             nLost, ncols, nrows);
    fflush(stderr);
  }

   if (nLost > 0) {
     _KLTSelectGoodFeatures(tc, img, ncols, nrows, fl, REPLACING_SOME);
   }
 
   if (KLT_verbose >= 1) {
     fprintf(stderr, "\n\t%d features replaced.\n",
             nLost - fl->nFeatures + KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
       fprintf(stderr, "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}
