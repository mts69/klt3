/*********************************************************************
 * pyramid.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>		/* malloc() ? */
#include <string.h>		/* memset() ? */
#include <math.h>		/* */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "pyramid.h"


/*********************************************************************
 *
 */

_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +	
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

     
  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid)  malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");
     
  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);

  /* Free structure */
  free(pyramid);
}


/*********************************************************************
 *
 */

// void _KLTComputePyramid(
//   _KLT_FloatImage img, 
//   _KLT_Pyramid pyramid,
//   float sigma_fact)
// {
//   _KLT_FloatImage currimg, tmpimg;
//   int ncols = img->ncols, nrows = img->nrows;
//   int subsampling = pyramid->subsampling;
//   int subhalf = subsampling / 2;
//   float sigma = subsampling * sigma_fact;  /* empirically determined */
//   int oldncols;
//   int i, x, y;
	
//   if (subsampling != 2 && subsampling != 4 && 
//       subsampling != 8 && subsampling != 16 && subsampling != 32)
//     KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
//              "be either 2, 4, 8, 16, or 32");

//   assert(pyramid->ncols[0] == img->ncols);
//   assert(pyramid->nrows[0] == img->nrows);

//   /* Copy original image to level 0 of pyramid */
//   memcpy(pyramid->img[0]->data, img->data, ncols*nrows*sizeof(float));

//   currimg = img;
//   for (i = 1 ; i < pyramid->nLevels ; i++)  {
//     tmpimg = _KLTCreateFloatImage(ncols, nrows);
//     _KLTComputeSmoothedImage(currimg, sigma, tmpimg);


//     /* Subsample */
//     oldncols = ncols;
//     ncols /= subsampling;  nrows /= subsampling;
//     for (y = 0 ; y < nrows ; y++)
//       for (x = 0 ; x < ncols ; x++)
//         pyramid->img[i]->data[y*ncols+x] = 
//           tmpimg->data[(subsampling*y+subhalf)*oldncols +
//                       (subsampling*x+subhalf)];

//     /* Reassign current image */
//     currimg = pyramid->img[i];
				
//     _KLTFreeFloatImage(tmpimg);
//   }
// }
 
void _KLTComputePyramid(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols;
  int i, x, y;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy original image to level 0 of pyramid */
  memcpy(pyramid->img[0]->data, img->data, ncols * nrows * sizeof(float));

  /* ================================================================
     Prepare for bulk smoothing on GPU
     ================================================================ */
  int count = pyramid->nLevels - 1;
  _KLT_FloatImage *input_imgs  = (_KLT_FloatImage *) malloc(count * sizeof(_KLT_FloatImage));
  _KLT_FloatImage *tmp_imgs    = (_KLT_FloatImage *) malloc(count * sizeof(_KLT_FloatImage));
  float *sigma_array           = (float *) malloc(count * sizeof(float));

  _KLT_FloatImage currimg = pyramid->img[0];
  int curr_ncols = ncols;
  int curr_nrows = nrows;

  /* Build input/output buffers for each pyramid level */
  for (i = 0; i < count; i++) {
    input_imgs[i] = (i == 0) ? currimg : pyramid->img[i];
    tmp_imgs[i]   = _KLTCreateFloatImage(curr_ncols, curr_nrows);
    sigma_array[i] = sigma;

    curr_ncols /= subsampling;
    curr_nrows /= subsampling;
  }

  /* ================================================================
     🚀 Run all smoothings at once on GPU
     ================================================================ */
  _KLTBulkComputeSmoothedImage(input_imgs, sigma_array, tmp_imgs, count);

  /* ================================================================
     Subsample and assign to pyramid levels
     ================================================================ */
  for (i = 1; i < pyramid->nLevels; i++) {
    _KLT_FloatImage smoothed = tmp_imgs[i - 1];
    oldncols = pyramid->ncols[i - 1];
    ncols = pyramid->ncols[i];
    nrows = pyramid->nrows[i];

    for (y = 0; y < nrows; y++) {
      for (x = 0; x < ncols; x++) {
        pyramid->img[i]->data[y * ncols + x] =
          smoothed->data[(subsampling * y + subhalf) * oldncols +
                         (subsampling * x + subhalf)];
      }
    }
  }

  /* ================================================================
     Cleanup
     ================================================================ */
  for (i = 0; i < count; i++) {
    _KLTFreeFloatImage(tmp_imgs[i]);
  }
  free(input_imgs);
  free(tmp_imgs);
  free(sigma_array);
}

/*********************************************************************
 * _KLTComputeDualPyramidBulk
 *
 * Builds two image pyramids (e.g., frame1 + frame2) using a single
 * bulk GPU call to _KLTBulkComputeSmoothedImage().
 *
 * This avoids two separate GPU kernel launches for each image pyramid.
 *********************************************************************/
void _KLTComputeDualPyramidBulk(
  _KLT_FloatImage img1,
  _KLT_FloatImage img2,
  _KLT_Pyramid pyramid1,
  _KLT_Pyramid pyramid2,
  float sigma_fact)
{
  const int nlevels = pyramid1->nLevels;
  const int subsampling = pyramid1->subsampling;

  assert(pyramid2->nLevels == nlevels);
  assert(pyramid2->subsampling == subsampling);
  assert(pyramid1->ncols[0] == img1->ncols && pyramid1->nrows[0] == img1->nrows);
  assert(pyramid2->ncols[0] == img2->ncols && pyramid2->nrows[0] == img2->nrows);

  // Copy base level images directly
  memcpy(pyramid1->img[0]->data, img1->data,
         img1->ncols * img1->nrows * sizeof(float));
  memcpy(pyramid2->img[0]->data, img2->data,
         img2->ncols * img2->nrows * sizeof(float));

  // -------------------------
  // Build all pyramid levels
  // -------------------------
  int totalLevels = (nlevels - 1) * 2;   // both pyramids except base level
  _KLT_FloatImage *input_imgs  = (_KLT_FloatImage*) malloc(sizeof(_KLT_FloatImage) * totalLevels);
  _KLT_FloatImage *smooth_imgs = (_KLT_FloatImage*) malloc(sizeof(_KLT_FloatImage) * totalLevels);
  float *sigma_array = (float*) malloc(sizeof(float) * totalLevels);

  int ncols1 = img1->ncols;
  int nrows1 = img1->nrows;
  int ncols2 = img2->ncols;
  int nrows2 = img2->nrows;
  float sigma1 = subsampling * sigma_fact;
  float sigma2 = subsampling * sigma_fact;
  int subhalf = subsampling / 2;

  int idx = 0;
  _KLT_FloatImage curr1 = img1;
  _KLT_FloatImage curr2 = img2;

  // Prepare both pyramids' smoothing jobs
  for (int level = 1; level < nlevels; ++level) {
    input_imgs[idx] = curr1;
    smooth_imgs[idx] = _KLTCreateFloatImage(ncols1, nrows1);
    sigma_array[idx++] = sigma1;

    input_imgs[idx] = curr2;
    smooth_imgs[idx] = _KLTCreateFloatImage(ncols2, nrows2);
    sigma_array[idx++] = sigma2;

    ncols1 /= subsampling;
    nrows1 /= subsampling;
    ncols2 /= subsampling;
    nrows2 /= subsampling;
  }

  // -------------------------
  // Perform GPU bulk smoothing
  // -------------------------
  _KLTBulkComputeSmoothedImage(input_imgs, sigma_array, smooth_imgs, totalLevels);

  // -------------------------
  // Subsample into pyramid levels
  // -------------------------
  ncols1 = img1->ncols;
  nrows1 = img1->nrows;
  ncols2 = img2->ncols;
  nrows2 = img2->nrows;

  idx = 0;
  for (int level = 1; level < nlevels; ++level) {
    // --- Pyramid 1 ---
    _KLT_FloatImage tmp1 = smooth_imgs[idx++];
    int oldncols1 = ncols1;
    ncols1 /= subsampling;
    nrows1 /= subsampling;
    for (int y = 0; y < nrows1; y++)
      for (int x = 0; x < ncols1; x++)
        pyramid1->img[level]->data[y*ncols1 + x] =
          tmp1->data[(subsampling*y + subhalf)*oldncols1 +
                     (subsampling*x + subhalf)];
    _KLTFreeFloatImage(tmp1);

    // --- Pyramid 2 ---
    _KLT_FloatImage tmp2 = smooth_imgs[idx++];
    int oldncols2 = ncols2;
    ncols2 /= subsampling;
    nrows2 /= subsampling;
    for (int y = 0; y < nrows2; y++)
      for (int x = 0; x < ncols2; x++)
        pyramid2->img[level]->data[y*ncols2 + x] =
          tmp2->data[(subsampling*y + subhalf)*oldncols2 +
                     (subsampling*x + subhalf)];
    _KLTFreeFloatImage(tmp2);
  }

  free(input_imgs);
  free(smooth_imgs);
  free(sigma_array);
}










