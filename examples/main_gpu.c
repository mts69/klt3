/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "pnmio.h"
#include "klt.h"



#ifndef
#define ULTRA_BATCH_SIZE 16
#endif


/* Define data directory - can be overridden at compile time */
#ifndef DATA_DIR
#define DATA_DIR "data/"
#endif

/* Define output directory - can be overridden at compile time */
#ifndef OUTPUT_DIR
#define OUTPUT_DIR "output/"
#endif

/* Define number of features - can be overridden at compile time */
#ifndef N_FEATURES
#define N_FEATURES 150
#endif

/* Define maximum frames - can be overridden at compile time */
#ifndef MAX_FRAMES
#define MAX_FRAMES 999999
#endif

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = N_FEATURES, nFrames = 10;
  int ncols, nrows;
  int i;
  
  clock_t start_time, end_time;
  double cpu_time_used;
  
  char cmd[256];
  sprintf(cmd, "ls %simg*.pgm 2>/dev/null | wc -l", DATA_DIR);
  FILE *fp = popen(cmd, "r");
  fscanf(fp, "%d", &nFrames);
  pclose(fp);

  printf("🚀 GPU Time: Starting KLT feature tracking...\n");
  start_time = clock();

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
  sprintf(fnamein, "%simg0.pgm", DATA_DIR);
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));
  
  //printf("🎯 GPU Time: Selecting features...\n");
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  //sprintf(fnameout, "%sfeat0.ppm", OUTPUT_DIR);
  //KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);




  KLT_PixelType ** frame_buffer = malloc(ULTRA_BATCH_SIZE * sizeof(KLT_PixelType*));
  for (int i = 0; i < ULTRA_BATCH_SIZE; i++) 
  {
    frame_buffer[i] = malloc(ncols * nrows * sizeof(KLT_PixelType));
  }

  KLT_PixelType **frame_buffer = malloc(ULTRA_BATCH_SIZE * sizeof(KLT_PixelType*));
  for (int i = 0; i < ULTRA_BATCH_SIZE; i++) 
  {
      frame_buffer[i] = malloc(ncols * nrows * sizeof(KLT_PixelType));
  }
    
  // Allocate pyramid arrays
  _KLT_Pyramid *pyramids = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
  _KLT_Pyramid *pyramids_gradx = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
  _KLT_Pyramid *pyramids_grady = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
  
  for (int i = 0; i < ULTRA_BATCH_SIZE; i++) 
  {
    pyramids[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    pyramids_gradx[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    pyramids_grady[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
  }
     







  
  //printf("🔄 GPU Time: Tracking features through %d frames...\n", nFrames-1);
  for (i = 1 ; i < nFrames && i < MAX_FRAMES ; i++)  {
    sprintf(fnamein, "%simg%d.pgm", DATA_DIR, i);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    //sprintf(fnameout, "%sfeat%d.ppm", OUTPUT_DIR, i);
    //KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }
  
  printf("💾 GPU Time: Saving feature table...\n");
  sprintf(fnameout, "%sfeatures.txt", OUTPUT_DIR);
  KLTWriteFeatureTable(ft, fnameout, "%5.1f");
  sprintf(fnameout, "%sfeatures.ft", OUTPUT_DIR);
  KLTWriteFeatureTable(ft, fnameout, NULL);
  
  end_time = clock();
  cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
  
  printf("⏱️  GPU Time: Total processing time: %.2f seconds\n", cpu_time_used);
  // _KLT_printConvolveTiming();  // Disabled for performance
  printf("✅ GPU Time: KLT feature tracking completed successfully!\n");
  
  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);
  
  printf("%.3f\n", cpu_time_used);

  return 0;
}
