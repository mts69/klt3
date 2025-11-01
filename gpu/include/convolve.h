/*********************************************************************
 * convolve.h
 *********************************************************************/

 #ifndef _CONVOLVE_H_
 #define _CONVOLVE_H_
 
 #include "klt.h"
 #include "klt_util.h"
 #include "pyramid.h"
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 void _KLTToFloatImage(
   KLT_PixelType *img,
   int ncols, int nrows,
   _KLT_FloatImage floatimg);
 
void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

// GPU-only version that keeps gradients on device (zero-copy)
void _KLTComputeGradientsGPU(
  _KLT_FloatImage img,
  float sigma,
  int ncols, int nrows,
  float **d_gradx_out,
  float **d_grady_out);

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);
 
void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

void _KLTBulkBuildPyramidsWithGradientsULTRA(
    KLT_PixelType **raw_images,        // [batch_size] input images
    _KLT_Pyramid *pyramids_out,        // [batch_size] smoothed pyramids
    _KLT_Pyramid *pyramids_gradx_out,  // [batch_size] gradient-X pyramids  
    _KLT_Pyramid *pyramids_grady_out,  // [batch_size] gradient-Y pyramids
    int batch_size,
    KLT_TrackingContext tc);

    
void _KLTCleanupUltraBuffers();

// Timing functions (DISABLED for performance)
// void _KLT_printConvolveTiming();
// void _KLT_resetConvolveTiming();

#ifdef __cplusplus
}
#endif

#endif
 