/*********************************************************************
 * convolve_cuda.cu - OPTIMIZED FOR AMPERE (RTX 3080, SM_86)
 * 
 * Ampere-Specific Optimizations:
 * 1. Ampere has 100KB shared memory (vs 64KB on Turing)
 * 2. 68 SMs with 8704 CUDA cores (vs 40 SMs/2560 cores on T4)
 * 3. 760 GB/s memory bandwidth (vs 320 GB/s on T4)
 * 4. 5MB L2 cache (vs 4MB on T4)
 * 5. Better async memory operations
 * 6. Separable convolution for efficiency
 * 7. Coalesced memory access patterns
 * 8. Shared memory tiling with bank conflict avoidance
 * 9. Persistent device buffers (3 buffers for gradient computation)
 * 10. Constant memory for convolution kernels
 * 11. Optimized for 32×8 thread blocks (256 threads, 8 warps)
 *********************************************************************/

 #include <assert.h>
 #include <math.h>
 #include <stdlib.h>
 #include <cuda_runtime.h>
 #include "base.h"
 #include "error.h"
 #include "convolve.h"
 #include "klt_util.h"
 
 #define MAX_KERNEL_WIDTH 71
 #define WARP_SIZE 32
 // Ampere RTX 3080: 68 SMs, 8704 cores - need MORE parallelism!
 // Larger blocks to maximize occupancy on Ampere
 #define BLOCK_DIM_X 32  // Full warp for coalescing
 #define BLOCK_DIM_Y 16  // 512 threads total (2× T4, better for Ampere's 68 SMs)
 #define MAX_KERNEL_SIZE 35
 
 #define CUDA_CHECK(call) \
   do { \
     cudaError_t err = call; \
     if (err != cudaSuccess) { \
       fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
       exit(EXIT_FAILURE); \
     } \
   } while(0)
 
 /*********************************************************************
  * Kernel Data Structures
  *********************************************************************/
 typedef struct {
   int width;
   float data[MAX_KERNEL_WIDTH];
 } ConvolutionKernel;
 
 static ConvolutionKernel gauss_kernel;
 static ConvolutionKernel gaussderiv_kernel;
 static float sigma_last = -10.0;
 
 // Constant memory for kernel (faster than global, cached)
 __constant__ float c_kernel[MAX_KERNEL_SIZE];
 
 /*********************************************************************
  * Persistent Device Buffers with Streams
  *********************************************************************/
 static struct {
  float *d_img1, *d_img2, *d_img_source;  // d_img_source for keeping original during gradient computation
   size_t allocated_size;
   cudaStream_t stream;
   bool initialized;
} g_gpu = {NULL, NULL, NULL, 0, NULL, false};
 
 static void ensure_gpu_buffers(size_t bytes) {
   if (!g_gpu.initialized) {
     CUDA_CHECK(cudaStreamCreate(&g_gpu.stream));
     // Ampere: Configure for maximum shared memory (up to 99KB per block)
     // Note: cudaDeviceSetSharedMemConfig is deprecated on modern GPUs
     // Ampere automatically manages shared memory/L1 cache partitioning
     
    // Ampere: L2 cache is automatically managed by the hardware
    // The 5MB L2 cache on RTX 3080 will cache frequently accessed data
    // No explicit configuration needed - Ampere is smart about caching!
     
     g_gpu.initialized = true;
   }
   
   if (bytes > g_gpu.allocated_size) {
     if (g_gpu.d_img1) {
       cudaFree(g_gpu.d_img1);
       cudaFree(g_gpu.d_img2);
      cudaFree(g_gpu.d_img_source);
     }
     CUDA_CHECK(cudaMalloc(&g_gpu.d_img1, bytes));
     CUDA_CHECK(cudaMalloc(&g_gpu.d_img2, bytes));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_img_source, bytes));
     g_gpu.allocated_size = bytes;
   }
 }
 
 /*********************************************************************
 * OPTIMIZED HORIZONTAL CONVOLUTION
  *********************************************************************/
 __global__ void convolveHoriz_Optimized(
   const float * __restrict__ imgin,
   float * __restrict__ imgout,
   int ncols, int nrows,
   int kernel_width)
 {
  const int radius = kernel_width / 2;
   const int tile_width = blockDim.x;
   const int tile_height = blockDim.y;
   
   // Shared memory with 8-byte padding for bank conflict avoidance
   // T4: 32 banks, 4-byte words → 8-byte padding = 2 words
   const int tile_stride = tile_width + 2 * radius + 8;  // +8 for padding
   extern __shared__ float s_tile[];
   
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int gx = blockIdx.x * tile_width + tx;
   const int gy = blockIdx.y * tile_height + ty;
   
   if (gy >= nrows) return;
   
   // ============ PHASE 1: COOPERATIVE TILE LOADING ============
   const int tile_start_col = blockIdx.x * tile_width - radius;
   const int total_cols = tile_width + 2 * radius;
   
   // Each warp loads one row cooperatively
   for (int row = ty; row < tile_height; row += tile_height) {
     int global_row = blockIdx.y * tile_height + row;
     if (global_row >= nrows) continue;
     
     const float* row_ptr = &imgin[global_row * ncols];
     float* s_row = &s_tile[row * tile_stride];
     
    // Load tile data: each thread handles multiple elements
    for (int local_col = tx; local_col < total_cols; local_col += tile_width) {
       int global_col = tile_start_col + local_col;
      s_row[local_col] = (global_col >= 0 && global_col < ncols) ? row_ptr[global_col] : 0.0f;
     }
   }
   __syncthreads();
   
   // ============ PHASE 2: COMPUTE CONVOLUTION ============
   if (gx >= ncols) return;
   
   // Zero boundary pixels
   if (gx < radius || gx >= ncols - radius) {
     imgout[gy * ncols + gx] = 0.0f;
     return;
   }
   
   // Convolution with aggressive unrolling
   float sum = 0.0f;
   int s_center = ty * tile_stride + tx + radius;
   
   // Unroll based on typical kernel sizes
   if (kernel_width <= 7) {
     #pragma unroll
     for (int k = 0; k < kernel_width; k++) {
       sum += s_tile[s_center - radius + k] * c_kernel[k];
     }
   } else if (kernel_width <= 15) {
     #pragma unroll 4
     for (int k = 0; k < kernel_width; k++) {
       sum += s_tile[s_center - radius + k] * c_kernel[k];
     }
   } else {
     #pragma unroll 2
     for (int k = 0; k < kernel_width; k++) {
       sum += s_tile[s_center - radius + k] * c_kernel[k];
     }
   }
   
   imgout[gy * ncols + gx] = sum;
 }
 
 /*********************************************************************
 * OPTIMIZED VERTICAL CONVOLUTION WITH COALESCED LOADS
 * 
 * Strategy:
 * 1. Load horizontally (COALESCED!) into shared memory
 * 2. Transpose in shared memory (fast!)
 * 3. Convolve on transposed layout
 * 4. Write results (already in correct orientation)
  *********************************************************************/
 __global__ void convolveVert_Optimized(
   const float * __restrict__ imgin,
   float * __restrict__ imgout,
   int ncols, int nrows,
   int kernel_width)
 {
   const int radius = kernel_width / 2;
   const int tile_width = blockDim.x;
   const int tile_height = blockDim.y;
   
  // Two shared memory tiles: one for coalesced load, one for transposed data
   const int tile_vert = tile_height + 2 * radius;
  const int load_stride = tile_width + 1;  // +1 to avoid bank conflicts
  const int conv_stride = tile_vert + 1;
  
  extern __shared__ float s_mem[];
  float* s_load = s_mem;                              // For coalesced loads
  float* s_conv = s_mem + tile_vert * load_stride;   // For transposed convolution
   
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int gx = blockIdx.x * tile_width + tx;
   const int gy = blockIdx.y * tile_height + ty;
   
   if (gx >= ncols) return;
   
  // ============ PHASE 1: COALESCED LOAD (each warp loads rows horizontally) ============
   const int tile_start_row = blockIdx.y * tile_height - radius;
   
   for (int local_row = ty; local_row < tile_vert; local_row += tile_height) {
     int global_row = tile_start_row + local_row;
     
     float val = 0.0f;
     if (global_row >= 0 && global_row < nrows && gx < ncols) {
      val = __ldg(&imgin[global_row * ncols + gx]);  // Coalesced read!
    }
    s_load[local_row * load_stride + tx] = val;
  }
  __syncthreads();
  
  // ============ PHASE 2: TRANSPOSE IN SHARED MEMORY ============
  // Transpose so vertical convolution becomes horizontal access
  for (int row = ty; row < tile_vert; row += tile_height) {
    s_conv[tx * conv_stride + row] = s_load[row * load_stride + tx];
   }
   __syncthreads();
   
  // ============ PHASE 3: COMPUTE CONVOLUTION (now horizontal in s_conv!) ============
   if (gy >= nrows) return;
   
   // Zero boundary pixels
   if (gy < radius || gy >= nrows - radius) {
     imgout[gy * ncols + gx] = 0.0f;
     return;
   }
   
  // Convolution - accessing s_conv horizontally (was vertical column in original)
   float sum = 0.0f;
  int s_col = ty + radius;
   
   if (kernel_width <= 7) {
     #pragma unroll
     for (int k = 0; k < kernel_width; k++) {
      sum += s_conv[tx * conv_stride + s_col - radius + k] * c_kernel[k];
     }
   } else if (kernel_width <= 15) {
     #pragma unroll 4
     for (int k = 0; k < kernel_width; k++) {
      sum += s_conv[tx * conv_stride + s_col - radius + k] * c_kernel[k];
     }
   } else {
     #pragma unroll 2
     for (int k = 0; k < kernel_width; k++) {
      sum += s_conv[tx * conv_stride + s_col - radius + k] * c_kernel[k];
     }
   }
   
   imgout[gy * ncols + gx] = sum;
 }
 
 /*********************************************************************
  * Host Wrapper Functions
  *********************************************************************/
 static void _convolveImageHoriz(
   _KLT_FloatImage imgin,
   ConvolutionKernel kernel,
   _KLT_FloatImage imgout)
 {
   const int ncols = imgin->ncols;
   const int nrows = imgin->nrows;
   const size_t nbytes = ncols * nrows * sizeof(float);
   
   ensure_gpu_buffers(nbytes);
   
  // Copy kernel to constant memory (reversed to match CPU convention)
  // CPU applies kernel in reverse: kernel.data[width-1] at left, kernel.data[0] at right
  // GPU applies forward: c_kernel[0] at left, c_kernel[width-1] at right
  float reversed_kernel[MAX_KERNEL_SIZE];
  for (int i = 0; i < kernel.width; i++) {
    reversed_kernel[i] = kernel.data[kernel.width - 1 - i];
  }
  CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel, 
     kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Copy input to device
   CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
     cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Launch configuration
   const int radius = kernel.width / 2;
   dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
   dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
             (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
   
  // Shared memory size (must match kernel calculation!)
  const int tile_stride = BLOCK_DIM_X + 2 * radius + 8;  // +8 for padding
   size_t shared_bytes = BLOCK_DIM_Y * tile_stride * sizeof(float);
   
  // Enable large shared memory if needed (Ampere supports up to 99KB)
  if (shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    // Note: Ampere GPUs have unified L1/shared memory - no carveout needed
  }
   
   convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
     g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, kernel.width);
   
   CUDA_CHECK(cudaGetLastError());
   
   // Copy result back
   CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img2, nbytes,
     cudaMemcpyDeviceToHost, g_gpu.stream));
   
   CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
   
   imgout->ncols = ncols;
   imgout->nrows = nrows;
 }
 
 static void _convolveImageVert(
   _KLT_FloatImage imgin,
   ConvolutionKernel kernel,
   _KLT_FloatImage imgout)
 {
   const int ncols = imgin->ncols;
   const int nrows = imgin->nrows;
   const size_t nbytes = ncols * nrows * sizeof(float);
   
   ensure_gpu_buffers(nbytes);
   
  // Copy kernel to constant memory (reversed to match CPU convention)
  float reversed_kernel[MAX_KERNEL_SIZE];
  for (int i = 0; i < kernel.width; i++) {
    reversed_kernel[i] = kernel.data[kernel.width - 1 - i];
  }
  CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
     kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Copy input to device
   CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
     cudaMemcpyHostToDevice, g_gpu.stream));
   
  // ============ VERTICAL CONVOLUTION ============
   const int radius = kernel.width / 2;
   dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
   dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
             (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
   
  // Calculate shared memory (two tiles: load + transposed)
   const int tile_vert = BLOCK_DIM_Y + 2 * radius;
  const int load_stride = BLOCK_DIM_X + 1;
  const int conv_stride = tile_vert + 1;
  size_t shared_bytes = (tile_vert * load_stride + BLOCK_DIM_X * conv_stride) * sizeof(float);
   
   if (shared_bytes > 48 * 1024) {
     CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
       cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
     // Note: Ampere GPUs have unified L1/shared memory - no carveout needed
   }
   
  // Vertical convolution
   convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
     g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, kernel.width);
   
   CUDA_CHECK(cudaGetLastError());
   
  // Copy result back to host
   CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img2, nbytes,
     cudaMemcpyDeviceToHost, g_gpu.stream));
   
   CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
   
   imgout->ncols = ncols;
   imgout->nrows = nrows;
 }
 
 /*********************************************************************
 * Separable Convolution - OPTIMIZED GPU VERSION
 * 
 * Keep data on GPU for both passes - only 2 CPU↔GPU transfers total!
  *********************************************************************/
 static void _convolveSeparate(
   _KLT_FloatImage imgin,
   ConvolutionKernel horiz_kernel,
   ConvolutionKernel vert_kernel,
   _KLT_FloatImage imgout)
 {
  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;
  const size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_gpu_buffers(nbytes);
  
  // ============ UPLOAD INPUT ONCE ============
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // ============ HORIZONTAL PASS (GPU → GPU) ============
  {
    // Copy kernel to constant memory (reversed)
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < horiz_kernel.width; i++) {
      reversed_kernel[i] = horiz_kernel.data[horiz_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      horiz_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    const int radius = horiz_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    const int tile_stride = BLOCK_DIM_X + 2 * radius + 8;
    size_t shared_bytes = BLOCK_DIM_Y * tile_stride * sizeof(float);
    
    // Debug: Print configuration
    if (shared_bytes > 99 * 1024) {
      fprintf(stderr, "ERROR: Shared memory too large: %zu bytes (max 99KB)\n", shared_bytes);
      fprintf(stderr, "  ncols=%d, nrows=%d, radius=%d, tile_stride=%d\n", 
              ncols, nrows, radius, tile_stride);
      return;
    }
    
    if (shared_bytes > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
      // Note: Ampere GPUs have unified L1/shared memory - no carveout needed
    }
    
    // d_img1 → d_img2
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, horiz_kernel.width);
    
    CUDA_CHECK(cudaGetLastError());
  }
  
  // ============ VERTICAL PASS (GPU → GPU) ============
  {
    // Copy kernel to constant memory (reversed)
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < vert_kernel.width; i++) {
      reversed_kernel[i] = vert_kernel.data[vert_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      vert_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    const int radius = vert_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    const int tile_vert = BLOCK_DIM_Y + 2 * radius;
    const int load_stride = BLOCK_DIM_X + 1;
    const int conv_stride = tile_vert + 1;
    // Two tiles: one for loading, one for transposed convolution
    size_t shared_bytes = (tile_vert * load_stride + BLOCK_DIM_X * conv_stride) * sizeof(float);
    
    if (shared_bytes > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
      // Note: Ampere GPUs have unified L1/shared memory - no carveout needed
    }
    
    // d_img2 → d_img1
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img2, g_gpu.d_img1, ncols, nrows, vert_kernel.width);
    
    CUDA_CHECK(cudaGetLastError());
  }
  
  // ============ DOWNLOAD RESULT ONCE ============
  CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img1, nbytes,
    cudaMemcpyDeviceToHost, g_gpu.stream));
  
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
  
  imgout->ncols = ncols;
  imgout->nrows = nrows;
 }
 
 /*********************************************************************
  * Kernel Computation (unchanged from original)
  *********************************************************************/
 static void _computeKernels(
   float sigma,
   ConvolutionKernel *gauss,
   ConvolutionKernel *gaussderiv)
 {
   const float factor = 0.01f;
   int i;
 
   assert(MAX_KERNEL_WIDTH % 2 == 1);
   assert(sigma >= 0.0);
 
   {
     const int hw = MAX_KERNEL_WIDTH / 2;
     float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));
   
     for (i = -hw; i <= hw; i++) {
       gauss->data[i+hw] = (float)exp(-i*i / (2*sigma*sigma));
       gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
     }
 
     gauss->width = MAX_KERNEL_WIDTH;
     for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; 
          i++, gauss->width -= 2);
     gaussderiv->width = MAX_KERNEL_WIDTH;
     for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; 
          i++, gaussderiv->width -= 2);
     if (gauss->width == MAX_KERNEL_WIDTH || 
         gaussderiv->width == MAX_KERNEL_WIDTH)
       KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
                "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
   }
 
   for (i = 0; i < gauss->width; i++)
     gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
   for (i = 0; i < gaussderiv->width; i++)
     gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
 
   {
     const int hw = gaussderiv->width / 2;
     float den;
       
     den = 0.0;
     for (i = 0; i < gauss->width; i++) den += gauss->data[i];
     for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;
     den = 0.0;
     for (i = -hw; i <= hw; i++) den -= i*gaussderiv->data[i+hw];
     for (i = -hw; i <= hw; i++) gaussderiv->data[i+hw] /= den;
   }
 
   sigma_last = sigma;
 }
 
 /*********************************************************************
  * Public API Functions
  *********************************************************************/
 void _KLTToFloatImage(
   KLT_PixelType *img,
   int ncols, int nrows,
   _KLT_FloatImage floatimg)
 {
   KLT_PixelType *ptrend = img + ncols*nrows;
   float *ptrout = floatimg->data;
 
   assert(floatimg->ncols >= ncols);
   assert(floatimg->nrows >= nrows);
 
   floatimg->ncols = ncols;
   floatimg->nrows = nrows;
 
   while (img < ptrend) *ptrout++ = (float)*img++;
 }
 
 void _KLTGetKernelWidths(
   float sigma,
   int *gauss_width,
   int *gaussderiv_width)
 {
   _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
   *gauss_width = gauss_kernel.width;
   *gaussderiv_width = gaussderiv_kernel.width;
 }
 
 void _KLTComputeGradients(
   _KLT_FloatImage img,
   float sigma,
   _KLT_FloatImage gradx,
   _KLT_FloatImage grady)
 {
   assert(gradx->ncols >= img->ncols);
   assert(gradx->nrows >= img->nrows);
   assert(grady->ncols >= img->ncols);
   assert(grady->nrows >= img->nrows);
 
   if (fabs(sigma - sigma_last) > 0.05)
     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
   
  const int ncols = img->ncols;
  const int nrows = img->nrows;
  const size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_gpu_buffers(nbytes);
  
  // ============ UPLOAD INPUT IMAGE ONCE TO SOURCE BUFFER ============
  // Ampere: Use async copy with stream for better overlap
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img_source, img->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // ============ COMPUTE GRADX: (gaussderiv_x * gauss_y) ============
  {
    // Horizontal pass with gaussderiv
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < gaussderiv_kernel.width; i++) {
      reversed_kernel[i] = gaussderiv_kernel.data[gaussderiv_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    int radius = gaussderiv_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    size_t shared_bytes = BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * radius + 8) * sizeof(float);
    
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img_source, g_gpu.d_img1, ncols, nrows, gaussderiv_kernel.width);
    
    // Vertical pass with gauss
    for (int i = 0; i < gauss_kernel.width; i++) {
      reversed_kernel[i] = gauss_kernel.data[gauss_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    radius = gauss_kernel.width / 2;
    grid = dim3((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    int tile_vert = BLOCK_DIM_Y + 2 * radius;
    shared_bytes = (tile_vert * (BLOCK_DIM_X + 1) + BLOCK_DIM_X * (tile_vert + 1)) * sizeof(float);
    
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, gauss_kernel.width);
    
    // Download gradx result asynchronously (don't wait!)
    CUDA_CHECK(cudaMemcpyAsync(gradx->data, g_gpu.d_img2, nbytes,
      cudaMemcpyDeviceToHost, g_gpu.stream));
  }
  
  // ============ COMPUTE GRADY: (gauss_x * gaussderiv_y) ============
  // Note: Original img is still in d_img_source - no re-upload needed!
  // Ampere: Can start grady computation while gradx is downloading (async overlap!)
  {
    // Horizontal pass with gauss
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < gauss_kernel.width; i++) {
      reversed_kernel[i] = gauss_kernel.data[gauss_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    int radius = gauss_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    size_t shared_bytes = BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * radius + 8) * sizeof(float);
    
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img_source, g_gpu.d_img1, ncols, nrows, gauss_kernel.width);
    
    // Vertical pass with gaussderiv
    for (int i = 0; i < gaussderiv_kernel.width; i++) {
      reversed_kernel[i] = gaussderiv_kernel.data[gaussderiv_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    radius = gaussderiv_kernel.width / 2;
    grid = dim3((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    int tile_vert_grady = BLOCK_DIM_Y + 2 * radius;
    shared_bytes = (tile_vert_grady * (BLOCK_DIM_X + 1) + BLOCK_DIM_X * (tile_vert_grady + 1)) * sizeof(float);
    
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, gaussderiv_kernel.width);
    
    // Download grady result
    CUDA_CHECK(cudaMemcpyAsync(grady->data, g_gpu.d_img2, nbytes,
      cudaMemcpyDeviceToHost, g_gpu.stream));
  }
  
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
  
  gradx->ncols = ncols;
  gradx->nrows = nrows;
  grady->ncols = ncols;
  grady->nrows = nrows;
 }
 
 void _KLTComputeSmoothedImage(
   _KLT_FloatImage img,
   float sigma,
   _KLT_FloatImage smooth)
 {
   assert(smooth->ncols >= img->ncols);
   assert(smooth->nrows >= img->nrows);
 
   if (fabs(sigma - sigma_last) > 0.05)
     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
 
  ensure_gpu_buffers(img->ncols * img->nrows * sizeof(float));
  
  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

/*********************************************************************/
/* GPU-ONLY VERSION: Keep gradients on device (ZERO-COPY!)          */
/* Returns device pointers - NO D2H transfer!                       */
/*********************************************************************/
void _KLTComputeGradientsGPU(
  _KLT_FloatImage img,
  float sigma,
  int ncols, int nrows,
  float **d_gradx_out,
  float **d_grady_out)
{
  static float sigma_last = -1.0f;
  static ConvolutionKernel gauss_kernel = {NULL, 0};
  static ConvolutionKernel gaussderiv_kernel = {NULL, 0};

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  
  const size_t nbytes = ncols * nrows * sizeof(float);
  ensure_gpu_buffers(nbytes);
  
  // Upload input image once
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img_source, img->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // ============ COMPUTE GRADX: (gaussderiv_x * gauss_y) ============
  {
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < gaussderiv_kernel.width; i++) {
      reversed_kernel[i] = gaussderiv_kernel.data[gaussderiv_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    int radius = gaussderiv_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    size_t shared_bytes = BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * radius + 8) * sizeof(float);
    
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img_source, g_gpu.d_img1, ncols, nrows, gaussderiv_kernel.width);
    
    // Vertical pass with gauss
    for (int i = 0; i < gauss_kernel.width; i++) {
      reversed_kernel[i] = gauss_kernel.data[gauss_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    radius = gauss_kernel.width / 2;
    int tile_vert = BLOCK_DIM_Y + 2 * radius;
    shared_bytes = (tile_vert * (BLOCK_DIM_X + 1) + BLOCK_DIM_X * (tile_vert + 1)) * sizeof(float);
    
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, gauss_kernel.width);
    
    // NO DOWNLOAD! Just return device pointer
    *d_gradx_out = g_gpu.d_img2;
  }
  
  // ============ COMPUTE GRADY: (gauss_x * gaussderiv_y) ============
  {
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < gauss_kernel.width; i++) {
      reversed_kernel[i] = gauss_kernel.data[gauss_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    int radius = gauss_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    size_t shared_bytes = BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * radius + 8) * sizeof(float);
    
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img_source, g_gpu.d_img1, ncols, nrows, gauss_kernel.width);
    
    // Vertical pass with gaussderiv
    for (int i = 0; i < gaussderiv_kernel.width; i++) {
      reversed_kernel[i] = gaussderiv_kernel.data[gaussderiv_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    radius = gaussderiv_kernel.width / 2;
    int tile_vert = BLOCK_DIM_Y + 2 * radius;
    shared_bytes = (tile_vert * (BLOCK_DIM_X + 1) + BLOCK_DIM_X * (tile_vert + 1)) * sizeof(float);
    
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img_source, ncols, nrows, gaussderiv_kernel.width);
    
    // NO DOWNLOAD! Just return device pointer
    *d_grady_out = g_gpu.d_img_source;
  }

  // Sync to ensure gradients are ready
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
}

// Cleanup function (call at program exit)
 void _KLTCleanupGPU() {
   if (g_gpu.initialized) {
     if (g_gpu.d_img1) cudaFree(g_gpu.d_img1);
     if (g_gpu.d_img2) cudaFree(g_gpu.d_img2);
    if (g_gpu.d_img_source) cudaFree(g_gpu.d_img_source);
     cudaStreamDestroy(g_gpu.stream);
     g_gpu.initialized = false;
   }
 }
