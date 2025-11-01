/*********************************************************************
 * klt_internal.h
 * 
 * Internal KLT functions needed for ULTRA mode
 * (These are normally private to the library)
 *********************************************************************/

#ifndef _KLT_INTERNAL_H_
#define _KLT_INTERNAL_H_

#include "base.h"
#include "klt.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************************************
 * Internal tracking function (from trackFeatures.c)
 *********************************************************************/
int _trackFeature(
    float x1, float y1,          /* location of window in first image */
    float *x2, float *y2,         /* starting location of window in second image */
    _KLT_FloatImage img1,
    _KLT_FloatImage gradx1,
    _KLT_FloatImage grady1,
    _KLT_FloatImage img2,
    _KLT_FloatImage gradx2,
    _KLT_FloatImage grady2,
    int width, int height,        /* size of window */
    float step_factor,            /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
    int max_iterations,
    float small_det,
    float th,                     /* displacement threshold */
    float max_residue,
    int lighting_insensitive);

/*********************************************************************
 * Boundary check function (from trackFeatures.c)
 *********************************************************************/
int _outOfBounds(
    float x, float y,
    int ncols, int nrows,
    int borderx, int bordery);

/*********************************************************************
 * Image conversion functions (from convolve.c or convolve_cuda.cu)
 *********************************************************************/
void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg);

/*********************************************************************
 * Smoothing function (from convolve.c or convolve_cuda.cu)
 *********************************************************************/
void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth);

/*********************************************************************
 * Gradient computation (from convolve.c or convolve_cuda.cu)
 *********************************************************************/
void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady);

/*********************************************************************
 * BULK functions for ULTRA mode (from convolve_cuda.cu)
 *********************************************************************/
void _KLTBulkComputeSmoothedImage(
    _KLT_FloatImage *img_array,
    float *sigma_array,
    _KLT_FloatImage *smooth_array,
    int count);

void _KLTBulkComputeGradients(
    _KLT_FloatImage *img_array,
    float sigma,
    _KLT_FloatImage *gradx_array,
    _KLT_FloatImage *grady_array,
    int count);

/*********************************************************************
 * ULTRA mode function (from convolve_cuda.cu)
 *********************************************************************/
void _KLTBulkBuildPyramidsWithGradientsULTRA(
    KLT_PixelType **raw_images,
    _KLT_Pyramid *pyramids_out,
    _KLT_Pyramid *pyramids_gradx_out,
    _KLT_Pyramid *pyramids_grady_out,
    int batch_size,
    KLT_TrackingContext tc);

void _KLTCleanupUltraBuffers(void);

#ifdef __cplusplus
}
#endif

#endif /* _KLT_INTERNAL_H_ */