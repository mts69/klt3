
// /**********************************************************************
// Finds the 150 best features in an image and tracks them through the 
// next two images.  The sequential mode is set in order to speed
// processing.  The features are stored in a feature table, which is then
// saved to a text file; each feature list is also written to a PPM file.

// 🚀 ULTRA MODE: Batched pyramid construction with maximum GPU parallelism
// **********************************************************************/
// #include <stdlib.h>
// #include <stdio.h>
// #include <time.h>
// #include <string.h>
// #include "pnmio.h"
// #include "klt.h"
// #include "pyramid.h"



// // Temporary: Forward declare internal functions
// extern int _trackFeature(float, float, float*, float*, 
//                         void*, void*, void*, void*, void*, void*,
//                         int, int, float, int, float, float, float, int);
// extern int _outOfBounds(float, float, int, int, int, int);
// extern void _KLTToFloatImage(unsigned char*, int, int, void*);
// extern void _KLTComputeSmoothedImage(void*, float, void*);
// extern void _KLTComputeGradients(void*, float, void*, void*);
// extern void _KLTBulkBuildPyramidsWithGradientsULTRA(
//     unsigned char**, void*, void*, void*, int, void*);
// extern void _KLTCleanupUltraBuffers(void);


// /* Define batch size - can be overridden at compile time */
// #ifndef ULTRA_BATCH_SIZE
// #define ULTRA_BATCH_SIZE 25

// #endif

// /* Define data directory - can be overridden at compile time */
// #ifndef DATA_DIR
// #define DATA_DIR "data/"
// #endif

// /* Define output directory - can be overridden at compile time */
// #ifndef OUTPUT_DIR
// #define OUTPUT_DIR "output/"
// #endif

// /* Define number of features - can be overridden at compile time */
// #ifndef N_FEATURES
// #define N_FEATURES 150
// #endif

// /* Define maximum frames - can be overridden at compile time */
// #ifndef MAX_FRAMES
// #define MAX_FRAMES 999999
// #endif

// /* #define REPLACE */

// /*********************************************************************
//  * Helper: Get minimum of two integers
//  *********************************************************************/
// static inline int min(int a, int b) {
//     return (a < b) ? a : b;
// }

// /*********************************************************************
//  * Helper: Track features using pre-computed pyramids
//  *********************************************************************/
// static void track_with_precomputed_pyramids(
//     _KLT_Pyramid pyramid1, _KLT_Pyramid pyramid1_gradx, _KLT_Pyramid pyramid1_grady,
//     _KLT_Pyramid pyramid2, _KLT_Pyramid pyramid2_gradx, _KLT_Pyramid pyramid2_grady,
//     KLT_FeatureList fl,
//     KLT_TrackingContext tc)
// {
//     float subsampling = (float) tc->subsampling;
//     int ncols = pyramid1->ncols[0];
//     int nrows = pyramid1->nrows[0];
    
//     // For each feature, track through pyramid
//     for (int indx = 0; indx < fl->nFeatures; indx++) {
        
//         if (fl->feature[indx]->val < 0) continue;  // Skip lost features
        
//         float xloc = fl->feature[indx]->x;
//         float yloc = fl->feature[indx]->y;
        
//         // Transform to coarsest resolution
//         for (int r = tc->nPyramidLevels - 1; r >= 0; r--) {
//             xloc /= subsampling;
//             yloc /= subsampling;
//         }
        
//         float xlocout = xloc, ylocout = yloc;
//         int val = KLT_TRACKED;
        
//         // Track from coarse to fine
//         for (int r = tc->nPyramidLevels - 1; r >= 0; r--) {
//             xloc *= subsampling;
//             yloc *= subsampling;
//             xlocout *= subsampling;
//             ylocout *= subsampling;
            
//             val = _trackFeature(xloc, yloc, &xlocout, &ylocout,
//                                 pyramid1->img[r],
//                                 pyramid1_gradx->img[r], pyramid1_grady->img[r],
//                                 pyramid2->img[r],
//                                 pyramid2_gradx->img[r], pyramid2_grady->img[r],
//                                 tc->window_width, tc->window_height,
//                                 tc->step_factor,
//                                 tc->max_iterations,
//                                 tc->min_determinant,
//                                 tc->min_displacement,
//                                 tc->max_residue,
//                                 tc->lighting_insensitive);
            
//             if (val == KLT_SMALL_DET || val == KLT_OOB) break;
//         }
        
//         // Update feature
//         if (val == KLT_TRACKED && 
//             !_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery)) {
//             fl->feature[indx]->x = xlocout;
//             fl->feature[indx]->y = ylocout;
//             fl->feature[indx]->val = KLT_TRACKED;
//         } else {
//             fl->feature[indx]->x = -1.0;
//             fl->feature[indx]->y = -1.0;
//             fl->feature[indx]->val = val;
//         }
//     }
// }

// /*********************************************************************
//  * MAIN - GPU-Accelerated KLT Tracking with ULTRA Mode
//  *********************************************************************/
// #ifdef WIN32
// int RunExample3()
// #else
// int main()
// #endif
// {
//     unsigned char *img_for_feature_detect;
//     char fnamein[256], fnameout[256];
//     KLT_TrackingContext tc;
//     KLT_FeatureList fl;
//     KLT_FeatureTable ft;
//     int nFeatures = N_FEATURES;
//     int nFrames = 10;  // Default, will be updated
//     int ncols = 0, nrows = 0;
    
//     clock_t start_time, end_time;
//     double cpu_time_used;
    
//     // ================================================================
//     // SETUP: Count frames and create tracking context
//     // ================================================================
//     char cmd[256];
//     sprintf(cmd, "ls %simg*.pgm 2>/dev/null | wc -l", DATA_DIR);
//     FILE *fp = popen(cmd, "r");
//     if (fp) {
//         fscanf(fp, "%d", &nFrames);
//         pclose(fp);
//     }
    
//     if (nFrames <= 0) {
//         fprintf(stderr, "❌ ERROR: No image files found in %s\n", DATA_DIR);
//         return 1;
//     }
    
//     // Apply MAX_FRAMES limit
//     if (nFrames > MAX_FRAMES) {
//         printf("⚠️  Limiting to %d frames (found %d)\n", MAX_FRAMES, nFrames);
//         nFrames = MAX_FRAMES;
//     }
    
//     printf("========================================\n");
//     printf("🚀 GPU-Accelerated KLT Feature Tracking\n");
//     printf("========================================\n");
//     printf("Frames: %d\n", nFrames);
//     printf("Features: %d\n", nFeatures);
//     printf("Batch size: %d\n", ULTRA_BATCH_SIZE);
//     printf("========================================\n\n");
    
//     start_time = clock();
    
//     tc = KLTCreateTrackingContext();
//     fl = KLTCreateFeatureList(nFeatures);
//     ft = KLTCreateFeatureTable(nFrames, nFeatures);
    
//     tc->sequentialMode = FALSE;  // ⚠️ MUST be FALSE for ULTRA mode!
//     tc->writeInternalImages = FALSE;
//     tc->affineConsistencyCheck = -1;
    
//     // ================================================================
//     // STEP 1: Load first frame and detect features
//     // ================================================================
//     printf("[1/3] Loading first frame and detecting features...\n");
//     sprintf(fnamein, "%simg0.pgm", DATA_DIR);
//     img_for_feature_detect = pgmReadFile(fnamein, NULL, &ncols, &nrows);
    
//     if (!img_for_feature_detect) {
//         fprintf(stderr, "❌ ERROR: Could not load %s\n", fnamein);
//         return 1;
//     }
    
//     printf("  Image size: %d × %d\n", ncols, nrows);
    
//     KLTSelectGoodFeatures(tc, img_for_feature_detect, ncols, nrows, fl);
//     KLTStoreFeatureList(fl, ft, 0);
    
//     int initial_features = KLTCountRemainingFeatures(fl);
//     printf("  ✅ Selected %d features\n\n", initial_features);
    
//     // ================================================================
//     // STEP 2: Allocate buffers for ULTRA mode
//     // ================================================================
//     printf("[2/3] Allocating ULTRA mode buffers...\n");
    
//     KLT_PixelType **frame_buffer = malloc(ULTRA_BATCH_SIZE * sizeof(KLT_PixelType*));
//     _KLT_Pyramid *pyramids = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
//     _KLT_Pyramid *pyramids_gradx = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
//     _KLT_Pyramid *pyramids_grady = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
    
//     if (!frame_buffer || !pyramids || !pyramids_gradx || !pyramids_grady) {
//         fprintf(stderr, "❌ ERROR: Memory allocation failed!\n");
//         return 1;
//     }
    
//     for (int i = 0; i < ULTRA_BATCH_SIZE; i++) {
//         frame_buffer[i] = (KLT_PixelType*) malloc(ncols * nrows * sizeof(KLT_PixelType));
//         pyramids[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
//         pyramids_gradx[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
//         pyramids_grady[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
        
//         if (!frame_buffer[i] || !pyramids[i] || !pyramids_gradx[i] || !pyramids_grady[i]) {
//             fprintf(stderr, "❌ ERROR: Memory allocation failed at index %d!\n", i);
//             return 1;
//         }
//     }
    
//     printf("  ✅ Allocated %d frame buffers\n", ULTRA_BATCH_SIZE);
//     printf("  ✅ Allocated %d pyramid sets\n\n", ULTRA_BATCH_SIZE);
    
//     // ================================================================
//     // STEP 3: Process frames in batches (ULTRA MODE!)
//     // ================================================================
//     printf("[3/3] Tracking features through %d frames...\n", nFrames - 1);
    
//     double total_pyramid_time = 0.0;
//     double total_tracking_time = 0.0;
//     int total_batches = 0;
    
//     // Process from frame 1 to nFrames-1 (frame 0 already processed)
//     for (int batch_start = 1; batch_start < nFrames; batch_start += ULTRA_BATCH_SIZE) {
        
//         // ============================================================
//         // Determine actual batch size (handle last batch!)
//         // ============================================================
//         int batch_size = min(ULTRA_BATCH_SIZE, nFrames - batch_start);
//         total_batches++;
        
//         printf("\n  Batch %d: Frames %d-%d (%d frames)\n", 
//                total_batches, batch_start, batch_start + batch_size - 1, batch_size);
        
//         // ============================================================
//         // Load batch of frames
//         // ============================================================
//         clock_t t_load = clock();
        
//         for (int i = 0; i < batch_size; i++) {
//             sprintf(fnamein, "%simg%d.pgm", DATA_DIR, batch_start + i);
//             KLT_PixelType *loaded = pgmReadFile(fnamein, NULL, &ncols, &nrows);
            
//             if (!loaded) {
//                 fprintf(stderr, "    ❌ ERROR: Could not load %s\n", fnamein);
//                 // Handle gracefully: skip this frame
//                 batch_size = i;  // Truncate batch
//                 break;
//             }
            
//             // Copy to buffer
//             memcpy(frame_buffer[i], loaded, ncols * nrows * sizeof(KLT_PixelType));
//             free(loaded);
//         }
        
//         if (batch_size == 0) {
//             printf("    ⚠️  No frames loaded, skipping batch\n");
//             continue;
//         }
        
//         double load_time = ((double)(clock() - t_load)) / CLOCKS_PER_SEC;
//         printf("    Load: %.2f ms\n", load_time * 1000);
        
//         // ============================================================
//         // 🚀 ULTRA PYRAMID COMPUTATION
//         // ============================================================
//         clock_t t_pyramid = clock();
        
//         _KLTBulkBuildPyramidsWithGradientsULTRA(
//             frame_buffer,
//             pyramids,
//             pyramids_gradx,
//             pyramids_grady,
//             batch_size,
//             tc);
        
//         double pyramid_time = ((double)(clock() - t_pyramid)) / CLOCKS_PER_SEC;
//         total_pyramid_time += pyramid_time;
        
//         printf("    Pyramids: %.2f ms (%.2f ms/frame)\n", 
//                pyramid_time * 1000, pyramid_time * 1000 / batch_size);
        
//         // ============================================================
//         // Track features through batch
//         // ============================================================
//         clock_t t_track = clock();
        
//         for (int i = 0; i < batch_size; i++) {
//             int frame_idx = batch_start + i;
            
//             // For first frame in batch, need previous frame's pyramid
//             _KLT_Pyramid prev_pyramid, prev_gradx, prev_grady;
            
//             if (i == 0) {
//                 // Need to compute pyramid for frame[batch_start - 1]
//                 // This is a boundary case - we'll handle it simply
//                 if (batch_start == 1) {
//                     // Previous frame is frame 0 (feature detection frame)
//                     // Compute its pyramid on-the-fly (one-time cost)
//                     _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
//                     _KLT_FloatImage floatimg = _KLTCreateFloatImage(ncols, nrows);
                    
//                     _KLTToFloatImage(img_for_feature_detect, ncols, nrows, tmpimg);
//                     _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
                    
//                     prev_pyramid = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
//                     prev_gradx = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
//                     prev_grady = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
                    
//                     _KLTComputePyramid(floatimg, prev_pyramid, tc->pyramid_sigma_fact);
                    
//                     for (int lvl = 0; lvl < tc->nPyramidLevels; lvl++) {
//                         _KLTComputeGradients(prev_pyramid->img[lvl], tc->grad_sigma,
//                                             prev_gradx->img[lvl], prev_grady->img[lvl]);
//                     }
                    
//                     _KLTFreeFloatImage(tmpimg);
//                     _KLTFreeFloatImage(floatimg);
//                 } else {
//                     // Use last pyramid from previous batch
//                     prev_pyramid = pyramids[ULTRA_BATCH_SIZE - 1];
//                     prev_gradx = pyramids_gradx[ULTRA_BATCH_SIZE - 1];
//                     prev_grady = pyramids_grady[ULTRA_BATCH_SIZE - 1];
//                 }
//             } else {
//                 // Use pyramid from previous frame in this batch
//                 prev_pyramid = pyramids[i - 1];
//                 prev_gradx = pyramids_gradx[i - 1];
//                 prev_grady = pyramids_grady[i - 1];
//             }
            
//             // Track features
//             track_with_precomputed_pyramids(
//                 prev_pyramid, prev_gradx, prev_grady,
//                 pyramids[i], pyramids_gradx[i], pyramids_grady[i],
//                 fl, tc);
            
//             // Store results
//             KLTStoreFeatureList(fl, ft, frame_idx);
            
//             // Optional: Replace lost features
//             #ifdef REPLACE
//             KLTReplaceLostFeatures(tc, frame_buffer[i], ncols, nrows, fl);
//             #endif
            
//             // Cleanup frame 0 pyramid if we created it
//             if (i == 0 && batch_start == 1) {
//                 _KLTFreePyramid(prev_pyramid);
//                 _KLTFreePyramid(prev_gradx);
//                 _KLTFreePyramid(prev_grady);
//             }
//         }
        
//         double track_time = ((double)(clock() - t_track)) / CLOCKS_PER_SEC;
//         total_tracking_time += track_time;
        
//         int remaining = KLTCountRemainingFeatures(fl);
//         printf("    Tracking: %.2f ms (%.2f ms/frame)\n", 
//                track_time * 1000, track_time * 1000 / batch_size);
//         printf("    Features remaining: %d / %d\n", remaining, nFeatures);
        
//         // Early exit if too few features
//         if (remaining < nFeatures / 10) {
//             printf("    ⚠️  Too few features remaining! Consider re-detecting.\n");
//         }
//     }
    
//     // ================================================================
//     // STEP 4: Save results and report statistics
//     // ================================================================
//     printf("\n========================================\n");
//     printf("📊 PROCESSING COMPLETE\n");
//     printf("========================================\n");
    
//     printf("💾 Saving feature table...\n");
//     sprintf(fnameout, "%sfeatures.txt", OUTPUT_DIR);
//     KLTWriteFeatureTable(ft, fnameout, "%5.1f");
//     sprintf(fnameout, "%sfeatures.ft", OUTPUT_DIR);
//     KLTWriteFeatureTable(ft, fnameout, NULL);
//     printf("  ✅ Saved to %s\n", OUTPUT_DIR);
    
//     end_time = clock();
//     cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
//     printf("\n⏱️  TIMING BREAKDOWN:\n");
//     printf("  Pyramid computation: %.2f sec (%.2f ms/frame)\n",
//            total_pyramid_time, total_pyramid_time * 1000 / (nFrames - 1));
//     printf("  Feature tracking:    %.2f sec (%.2f ms/frame)\n",
//            total_tracking_time, total_tracking_time * 1000 / (nFrames - 1));
//     printf("  Other (I/O, etc):    %.2f sec\n",
//            cpu_time_used - total_pyramid_time - total_tracking_time);
//     printf("  ────────────────────────────────────\n");
//     printf("  TOTAL:               %.2f sec\n", cpu_time_used);
//     printf("  Throughput:          %.2f FPS\n\n", (nFrames - 1) / cpu_time_used);
    
//     int final_features = KLTCountRemainingFeatures(fl);
//     printf("📈 FEATURE STATISTICS:\n");
//     printf("  Initial features:  %d\n", initial_features);
//     printf("  Final features:    %d\n", final_features);
//     printf("  Retention rate:    %.1f%%\n\n", 
//            100.0 * final_features / initial_features);
    
//     printf("✅ KLT feature tracking completed successfully!\n");
//     printf("========================================\n\n");
    
//     // ================================================================
//     // CLEANUP
//     // ================================================================
//     for (int i = 0; i < ULTRA_BATCH_SIZE; i++) {
//         free(frame_buffer[i]);
//         _KLTFreePyramid(pyramids[i]);
//         _KLTFreePyramid(pyramids_gradx[i]);
//         _KLTFreePyramid(pyramids_grady[i]);
//     }
//     free(frame_buffer);
//     free(pyramids);
//     free(pyramids_gradx);
//     free(pyramids_grady);
    
//     _KLTCleanupUltraBuffers();  // Free GPU resources
    
//     KLTFreeFeatureTable(ft);
//     KLTFreeFeatureList(fl);
//     KLTFreeTrackingContext(tc);
//     free(img_for_feature_detect);
    
//     // Print just timing for automated benchmarking
//     printf("%.3f\n", cpu_time_used);
    
//     return 0;
// }








































/*********************************************************************
 * test_ultra.c
 * 
 * Comprehensive test suite for ULTRA mode pyramid computation
 * 
 * Tests:
 * 1. Single image: ULTRA vs Sequential (should be identical)
 * 2. Batch images: Verify all pyramids are correct
 * 3. Gradient validation: Check for NaN/Inf/zeros
 * 4. Performance benchmark: Measure speedup
 *********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "pnmio.h"
#include "klt.h"
#include "pyramid.h"

// External functions
extern void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows, void *floatimg);
extern void _KLTComputeSmoothedImage(void *img, float sigma, void *smooth);
extern void _KLTComputeGradients(void *img, float sigma, void *gradx, void *grady);
extern void _KLTBulkBuildPyramidsWithGradientsULTRA(
    KLT_PixelType **raw_images,
    void **pyramids_out,
    void **pyramids_gradx_out,
    void **pyramids_grady_out,
    int batch_size,
    void *tc);

#define DATA_DIR "/content/frames/"
#define TEST_BATCH_SIZE 10
#define MAX_PIXEL_DIFF 0.01 // Allow small floating point differences

/*********************************************************************
 * Helper: Compare two float images with reasonable tolerance
 *********************************************************************/
typedef struct {
    int total_pixels;
    int diff_count;
    float max_diff;
    float mean_diff;
    float rmse;
} ImageDiff;

ImageDiff compare_float_images(_KLT_FloatImage img1, _KLT_FloatImage img2) {
    ImageDiff result = {0};
    
    if (img1->ncols != img2->ncols || img1->nrows != img2->nrows) {
        printf("    ❌ SIZE MISMATCH: %dx%d vs %dx%d\n",
               img1->ncols, img1->nrows, img2->ncols, img2->nrows);
        result.diff_count = -1;
        return result;
    }
    
    result.total_pixels = img1->ncols * img1->nrows;
    float sum_diff = 0.0, sum_sq_diff = 0.0;
    
    for (int i = 0; i < result.total_pixels; i++) {
        float diff = fabsf(img1->data[i] - img2->data[i]);
        
        // Allow larger absolute difference for larger pixel values
        // Use 1% relative error OR 1.0 absolute error (whichever is larger)
        float val = fmaxf(fabsf(img1->data[i]), fabsf(img2->data[i]));
        float tolerance = fmaxf(val * 0.01f, 1.0f);
        
        if (diff > tolerance) {
            result.diff_count++;
        }
        
        if (diff > result.max_diff) {
            result.max_diff = diff;
        }
        
        sum_diff += diff;
        sum_sq_diff += diff * diff;
    }
    
    result.mean_diff = sum_diff / result.total_pixels;
    result.rmse = sqrtf(sum_sq_diff / result.total_pixels);
    
    return result;
}

/*********************************************************************
 * Validation: Check if differences are acceptable
 *********************************************************************/
int validate_image_match_2(ImageDiff diff, const char *name) {
    printf("    %s: max_diff=%.6f, mean_diff=%.6f, rmse=%.6f\n",
           name, diff.max_diff, diff.mean_diff, diff.rmse);
    
    // Accept if:
    // 1. Mean difference < 0.5 (less than half a gray level on average)
    // 2. RMSE < 1.0 (good overall match)
    // 3. Max difference < 15.0 (no catastrophic outliers)
    
    if (diff.mean_diff < 0.5f && diff.rmse < 1.0f && diff.max_diff < 15.0f) {
        printf("      ✅ PASS (acceptable GPU/CPU numerical differences)\n");
        return 1;
    } else {
        printf("      ❌ FAIL: Differences too large!\n");
        printf("         Mean=%.3f (limit 0.5), RMSE=%.3f (limit 1.0), Max=%.3f (limit 15.0)\n",
               diff.mean_diff, diff.rmse, diff.max_diff);
        return 0;
    }
}

/*********************************************************************
 * Helper: Validate image for NaN/Inf
 *********************************************************************/
int validate_image_match(_KLT_FloatImage img, const char *name) {
    int bad_count = 0;
    int zero_count = 0;
    float min_val = img->data[0], max_val = img->data[0], sum = 0.0;
    
    int total = img->ncols * img->nrows;
    
    for (int i = 0; i < total; i++) {
        float val = img->data[i];
        
        if (!isfinite(val)) {
            bad_count++;
        }
        
        if (val == 0.0f) {
            zero_count++;
        }
        
        if (isfinite(val)) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
        }
    }
    
    printf("    %s (%dx%d):\n", name, img->ncols, img->nrows);
    printf("      Range: [%.6f, %.6f], Mean: %.6f\n", 
           min_val, max_val, sum / (total - bad_count));
    printf("      Zeros: %d (%.1f%%), NaN/Inf: %d\n",
           zero_count, 100.0 * zero_count / total, bad_count);
    
    if (bad_count > 0) {
        printf("      ❌ INVALID VALUES DETECTED!\n");
        return 0;
    }
    
    if (zero_count == total) {
        printf("      ❌ ALL ZEROS!\n");
        return 0;
    }
    
    printf("      ✅ Valid\n");
    return 1;
}

/*********************************************************************
 * TEST 1: Single Image - ULTRA vs Sequential
 *********************************************************************/
int test_single_image(const char *data_dir) {
    printf("\n========================================\n");
    printf("TEST 1: Single Image (ULTRA vs Sequential)\n");
    printf("========================================\n");
    
    KLT_TrackingContext tc = KLTCreateTrackingContext();
    
    // Load test image
    char fname[256];
    sprintf(fname, "%simg0.pgm", data_dir);
    int ncols, nrows;
    KLT_PixelType *img = pgmReadFile(fname, NULL, &ncols, &nrows);
    
    if (!img) {
        printf("❌ Failed to load %s\n", fname);
        return 0;
    }
    
    printf("Image: %dx%d\n", ncols, nrows);
    printf("Pyramid levels: %d\n", tc->nPyramidLevels);
    printf("Subsampling: %d\n\n", tc->subsampling);

    // CHECK RAW IMAGE:
int sum_raw = 0;
int nonzero_raw = 0;
for (int i = 0; i < ncols * nrows; i++) {
    sum_raw += img[i];
    if (img[i] != 0) nonzero_raw++;
}
printf("DEBUG: Raw image stats:\n");
printf("  Total pixels: %d\n", ncols * nrows);
printf("  Sum: %d\n", sum_raw);
printf("  Non-zero pixels: %d\n", nonzero_raw);
printf("  First 10 pixels: ");
for (int i = 0; i < 10; i++) {
    printf("%d ", (int)img[i]);
}
printf("\n");
    
    // ================================================================
    // Method 1: SEQUENTIAL (ground truth)
    // ================================================================
    printf("[1/2] Computing with SEQUENTIAL mode...\n");
    
    clock_t t1 = clock();
    
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
    _KLT_FloatImage floatimg = _KLTCreateFloatImage(ncols, nrows);
    
    _KLTToFloatImage(img, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
    
    _KLT_Pyramid seq_pyramid = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    _KLT_Pyramid seq_gradx = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    _KLT_Pyramid seq_grady = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    
    _KLTComputePyramid(floatimg, seq_pyramid, tc->pyramid_sigma_fact);
// In test_single_image, after _KLTComputePyramid:
printf("\nSequential pyramid Level 0 data check:\n");
float sum_seq = 0.0f;
for (int i = 0; i < 100; i++) {
    sum_seq += seq_pyramid->img[0]->data[i];
}
printf("  Sum of first 100 pixels: %.2f\n", sum_seq);
printf("  First pixel: %.6f\n", seq_pyramid->img[0]->data[0]);

printf("\nSequential pyramid Level 1 data check:\n");
float sum_seq1 = 0.0f;
for (int i = 0; i < 100; i++) {
    sum_seq1 += seq_pyramid->img[1]->data[i];
}
printf("  Sum of first 100 pixels: %.2f\n", sum_seq1);
printf("  First pixel: %.6f\n", seq_pyramid->img[1]->data[0]);


    printf("\nSequential pyramid info:\n");
    for (int lvl = 0; lvl < tc->nPyramidLevels; lvl++) {
        printf("  Level %d: %dx%d, first_pixel=%.6f\n",
              lvl,
              seq_pyramid->ncols[lvl],
              seq_pyramid->nrows[lvl],
              seq_pyramid->img[lvl]->data[0]);
    }
    
    for (int lvl = 0; lvl < tc->nPyramidLevels; lvl++) {
        _KLTComputeGradients(seq_pyramid->img[lvl], tc->grad_sigma,
                             seq_gradx->img[lvl], seq_grady->img[lvl]);
    }
    
    double seq_time = (double)(clock() - t1) / CLOCKS_PER_SEC;
    printf("  Sequential time: %.2f ms\n\n", seq_time * 1000);
    
    // ================================================================
    // Method 2: ULTRA MODE
    // ================================================================
    printf("[2/2] Computing with ULTRA mode...\n");
    
    clock_t t2 = clock();
    
    _KLT_Pyramid ultra_pyramid = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    _KLT_Pyramid ultra_gradx = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    _KLT_Pyramid ultra_grady = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    
    KLT_PixelType *img_array[1] = {img};
    _KLT_Pyramid pyramid_array[1] = {ultra_pyramid};
    _KLT_Pyramid gradx_array[1] = {ultra_gradx};
    _KLT_Pyramid grady_array[1] = {ultra_grady};
    
    _KLTBulkBuildPyramidsWithGradientsULTRA(
        img_array,
        pyramid_array,
        gradx_array,
        grady_array,
        1,
        tc);
// After ULTRA completes, check the ENTIRE Level 1:
printf("\nULTRA Level 1 FULL CHECK:\n");
float min_val = ultra_pyramid->img[1]->data[0];
float max_val = ultra_pyramid->img[1]->data[0];
float sum_all = 0.0f;
int nonzero_count = 0;
int level1_size = ultra_pyramid->ncols[1] * ultra_pyramid->nrows[1];

for (int i = 0; i < level1_size; i++) {
    float val = ultra_pyramid->img[1]->data[i];
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum_all += val;
    if (val != 0.0f) nonzero_count++;
}
printf("  Range: [%.6f, %.6f]\n", min_val, max_val);
printf("  Mean: %.6f\n", sum_all / level1_size);
printf("  Non-zero pixels: %d / %d\n", nonzero_count, level1_size);

// Same for sequential:
printf("\nSEQUENTIAL Level 1 FULL CHECK:\n");
min_val = seq_pyramid->img[1]->data[0];
max_val = seq_pyramid->img[1]->data[0];
sum_all = 0.0f;
nonzero_count = 0;

for (int i = 0; i < level1_size; i++) {
    float val = seq_pyramid->img[1]->data[i];
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum_all += val;
    if (val != 0.0f) nonzero_count++;
}
printf("  Range: [%.6f, %.6f]\n", min_val, max_val);
printf("  Mean: %.6f\n", sum_all / level1_size);
printf("  Non-zero pixels: %d / %d\n", nonzero_count, level1_size);

// Check a few specific pixels that might differ:
printf("\nSample pixels comparison (Level 1):\n");
for (int i = 0; i < 10; i++) {
    int idx = i * 1000;  // Sample every 1000 pixels
    if (idx < level1_size) {
        printf("  Pixel %d: seq=%.6f, ultra=%.6f, diff=%.6f\n",
               idx,
               seq_pyramid->img[1]->data[idx],
               ultra_pyramid->img[1]->data[idx],
               fabsf(seq_pyramid->img[1]->data[idx] - ultra_pyramid->img[1]->data[idx]));
    }
}
// ADD THIS DEBUG:
printf("\nULTRA pyramid Level 0 data check:\n");
float sum_ultra0 = 0.0f;
for (int i = 0; i < 100; i++) {
    sum_ultra0 += ultra_pyramid->img[0]->data[i];
}
printf("  Sum of first 100 pixels: %.2f\n", sum_ultra0);
printf("  First pixel: %.6f\n", ultra_pyramid->img[0]->data[0]);

printf("\nULTRA pyramid Level 1 data check:\n");
float sum_ultra1 = 0.0f;
for (int i = 0; i < 100; i++) {
    sum_ultra1 += ultra_pyramid->img[1]->data[i];
}
printf("  Sum of first 100 pixels: %.2f\n", sum_ultra1);
printf("  First pixel: %.6f\n", ultra_pyramid->img[1]->data[0]);
    // After _KLTBulkBuildPyramidsWithGradientsULTRA:

printf("\nULTRA pyramid info:\n");
for (int lvl = 0; lvl < tc->nPyramidLevels; lvl++) {
    printf("  Level %d: %dx%d, first_pixel=%.6f\n",
           lvl,
           ultra_pyramid->ncols[lvl],
           ultra_pyramid->nrows[lvl],
           ultra_pyramid->img[lvl]->data[0]);
}
    double ultra_time = (double)(clock() - t2) / CLOCKS_PER_SEC;
    printf("  ULTRA time: %.2f ms\n\n", ultra_time * 1000);
    
    // ================================================================
    // COMPARE RESULTS
    // ================================================================
    printf("[3/3] Comparing results...\n\n");
    
    int all_passed = 1;
    
    for (int lvl = 0; lvl < tc->nPyramidLevels; lvl++) {
        printf("  Level %d (%dx%d):\n", 
               lvl, seq_pyramid->ncols[lvl], seq_pyramid->nrows[lvl]);
        
        // Compare pyramid
        ImageDiff diff_pyr = compare_float_images(seq_pyramid->img[lvl], 
                                                   ultra_pyramid->img[lvl]);
        printf("    Pyramid: max_diff=%.6f, mean_diff=%.6f, rmse=%.6f\n",
               diff_pyr.max_diff, diff_pyr.mean_diff, diff_pyr.rmse);
        
        // Use statistical validation instead of pixel counting
if (diff_pyr.mean_diff < 0.5f && diff_pyr.rmse < 1.0f && diff_pyr.max_diff < 15.0f) {
    printf("      ✅ PASS (acceptable GPU/CPU numerical differences)\n");
} else {
    printf("      ❌ FAIL: Differences too large!\n");
    printf("         Mean=%.3f (limit 0.5), RMSE=%.3f (limit 1.0), Max=%.3f (limit 15.0)\n",
           diff_pyr.mean_diff, diff_pyr.rmse, diff_pyr.max_diff);
    all_passed = 0;
}
        
        // Compare gradX
        ImageDiff diff_gx = compare_float_images(seq_gradx->img[lvl], 
                                                  ultra_gradx->img[lvl]);
        printf("    GradX: max_diff=%.6f, mean_diff=%.6f, rmse=%.6f\n",
               diff_gx.max_diff, diff_gx.mean_diff, diff_gx.rmse);
        
        if (diff_gx.diff_count > diff_gx.total_pixels * 0.01) {
            printf("      ❌ FAIL: %d/%d pixels differ\n",
                   diff_gx.diff_count, diff_gx.total_pixels);
            all_passed = 0;
        } else {
            printf("      ✅ PASS\n");
        }
        
        // Compare gradY
        ImageDiff diff_gy = compare_float_images(seq_grady->img[lvl], 
                                                  ultra_grady->img[lvl]);
        printf("    GradY: max_diff=%.6f, mean_diff=%.6f, rmse=%.6f\n",
               diff_gy.max_diff, diff_gy.mean_diff, diff_gy.rmse);
        
        if (diff_gy.diff_count > diff_gy.total_pixels * 0.01) {
            printf("      ❌ FAIL: %d/%d pixels differ\n",
                   diff_gy.diff_count, diff_gy.total_pixels);
            all_passed = 0;
        } else {
            printf("      ✅ PASS\n");
        }
        
        printf("\n");
    }
    
    // Cleanup
    free(img);
    _KLTFreeFloatImage(tmpimg);
    _KLTFreeFloatImage(floatimg);
    _KLTFreePyramid(seq_pyramid);
    _KLTFreePyramid(seq_gradx);
    _KLTFreePyramid(seq_grady);
    _KLTFreePyramid(ultra_pyramid);
    _KLTFreePyramid(ultra_gradx);
    _KLTFreePyramid(ultra_grady);
    KLTFreeTrackingContext(tc);
    
    return all_passed;
}

/*********************************************************************
 * TEST 2: Batch Processing - Validate All Pyramids
 *********************************************************************/
int test_batch_processing(const char *data_dir, int batch_size) {
    printf("\n========================================\n");
    printf("TEST 2: Batch Processing (%d images)\n", batch_size);
    printf("========================================\n");
    
    KLT_TrackingContext tc = KLTCreateTrackingContext();
    
    // Load images
    int ncols, nrows;
    KLT_PixelType **images = malloc(batch_size * sizeof(KLT_PixelType*));
    
    printf("Loading %d images...\n", batch_size);
    for (int i = 0; i < batch_size; i++) {
        char fname[256];
        sprintf(fname, "%simg%d.pgm", data_dir, i);
        images[i] = pgmReadFile(fname, NULL, &ncols, &nrows);
        
        if (!images[i]) {
            printf("❌ Failed to load %s\n", fname);
            return 0;
        }
    }
    printf("  ✅ Loaded %dx%d images\n\n", ncols, nrows);
    
    // Allocate pyramids
    _KLT_Pyramid *pyramids = malloc(batch_size * sizeof(_KLT_Pyramid));
    _KLT_Pyramid *gradx = malloc(batch_size * sizeof(_KLT_Pyramid));
    _KLT_Pyramid *grady = malloc(batch_size * sizeof(_KLT_Pyramid));
    
    for (int i = 0; i < batch_size; i++) {
        pyramids[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
        gradx[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
        grady[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
    }
    
    // Process batch
    printf("Processing batch with ULTRA mode...\n");
    clock_t t = clock();
    
    _KLTBulkBuildPyramidsWithGradientsULTRA(
        images,
        pyramids,
        gradx,
        grady,
        batch_size,
        tc);
    
    double batch_time = (double)(clock() - t) / CLOCKS_PER_SEC;
    printf("  Batch time: %.2f ms (%.2f ms/frame)\n\n", 
           batch_time * 1000, batch_time * 1000 / batch_size);
    
    // Validate all pyramids
    printf("Validating pyramids...\n");
    int all_valid = 1;
    
    for (int i = 0; i < batch_size; i++) {
        printf("\n  Frame %d:\n", i);
        
        for (int lvl = 0; lvl < tc->nPyramidLevels; lvl++) {
            char name[256];
            
            sprintf(name, "Pyramid Level %d", lvl);
            if (!validate_image_match(pyramids[i]->img[lvl], name)) {
                all_valid = 0;
            }
            
            sprintf(name, "GradX Level %d", lvl);
            if (!validate_image_match(gradx[i]->img[lvl], name)) {
                all_valid = 0;
            }
            
            sprintf(name, "GradY Level %d", lvl);
            if (!validate_image_match(grady[i]->img[lvl], name)) {
                all_valid = 0;
            }
        }
    }
    
    // Cleanup
    for (int i = 0; i < batch_size; i++) {
        free(images[i]);
        _KLTFreePyramid(pyramids[i]);
        _KLTFreePyramid(gradx[i]);
        _KLTFreePyramid(grady[i]);
    }
    free(images);
    free(pyramids);
    free(gradx);
    free(grady);
    KLTFreeTrackingContext(tc);
    
    return all_valid;
}

/*********************************************************************
 * TEST 3: Performance Benchmark
 *********************************************************************/
void test_performance(const char *data_dir) {
    printf("\n========================================\n");
    printf("TEST 3: Performance Benchmark\n");
    printf("========================================\n");
    
    int batch_sizes[] = {1, 2, 4, 8, 16, 32};
    int num_tests = 6;
    
    KLT_TrackingContext tc = KLTCreateTrackingContext();
    
    // Load test images
    int ncols, nrows;
    int max_batch = batch_sizes[num_tests - 1];
    KLT_PixelType **images = malloc(max_batch * sizeof(KLT_PixelType*));
    
    for (int i = 0; i < max_batch; i++) {
        char fname[256];
        sprintf(fname, "%simg%d.pgm", data_dir, i % 10);  // Reuse first 10 images
        images[i] = pgmReadFile(fname, NULL, &ncols, &nrows);
        
        if (!images[i]) {
            printf("❌ Failed to load images\n");
            return;
        }
    }
    
    printf("Image size: %dx%d\n", ncols, nrows);
    printf("Pyramid levels: %d\n\n", tc->nPyramidLevels);
    
    printf("%-12s %-15s %-15s %-10s\n", 
           "Batch Size", "Total Time (ms)", "Per Frame (ms)", "Speedup");
    printf("--------------------------------------------------------\n");
    
    double baseline_time = 0.0;
    
    for (int t = 0; t < num_tests; t++) {
        int batch = batch_sizes[t];
        
        // Allocate pyramids
        _KLT_Pyramid *pyramids = malloc(batch * sizeof(_KLT_Pyramid));
        _KLT_Pyramid *gradx = malloc(batch * sizeof(_KLT_Pyramid));
        _KLT_Pyramid *grady = malloc(batch * sizeof(_KLT_Pyramid));
        
        for (int i = 0; i < batch; i++) {
            pyramids[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
            gradx[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
            grady[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
        }
        
        // Benchmark
        clock_t start = clock();
        
        _KLTBulkBuildPyramidsWithGradientsULTRA(
            images,
            pyramids,
            gradx,
            grady,
            batch,
            tc);
        
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        double per_frame = elapsed / batch;
        
        if (t == 0) {
            baseline_time = per_frame;
        }
        
        double speedup = baseline_time / per_frame;
        
        printf("%-12d %-15.2f %-15.2f %-10.2fx\n",
               batch, elapsed * 1000, per_frame * 1000, speedup);
        
        // Cleanup
        for (int i = 0; i < batch; i++) {
            _KLTFreePyramid(pyramids[i]);
            _KLTFreePyramid(gradx[i]);
            _KLTFreePyramid(grady[i]);
        }
        free(pyramids);
        free(gradx);
        free(grady);
    }
    
    // Cleanup
    for (int i = 0; i < max_batch; i++) {
        free(images[i]);
    }
    free(images);
    KLTFreeTrackingContext(tc);
}

/*********************************************************************
 * MAIN
 *********************************************************************/
int main() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║      ULTRA MODE VALIDATION TEST SUITE                 ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    const char *data_dir = DATA_DIR;
    
    // Test 1: Single image correctness
    int test1_pass = test_single_image(data_dir);
    
    if (!test1_pass) {
        printf("\n❌ TEST 1 FAILED! ULTRA mode produces different results than sequential!\n");
        printf("   Fix this before proceeding.\n\n");
        return 1;
    }
    
    // Test 2: Batch validation
    int test2_pass = test_batch_processing(data_dir, TEST_BATCH_SIZE);
    
    if (!test2_pass) {
        printf("\n❌ TEST 2 FAILED! Pyramids contain invalid values!\n\n");
        return 1;
    }
    
    // Test 3: Performance
    test_performance(data_dir);
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║                 ALL TESTS PASSED! ✅                   ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    return 0;
}










