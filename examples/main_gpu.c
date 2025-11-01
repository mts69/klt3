
/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.

🚀 ULTRA MODE: Batched pyramid construction with maximum GPU parallelism
**********************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "pnmio.h"
#include "klt.h"
#include "pyramid.h"



// Temporary: Forward declare internal functions
extern int _trackFeature(float, float, float*, float*, 
                        void*, void*, void*, void*, void*, void*,
                        int, int, float, int, float, float, float, int);
extern int _outOfBounds(float, float, int, int, int, int);
extern void _KLTToFloatImage(unsigned char*, int, int, void*);
extern void _KLTComputeSmoothedImage(void*, float, void*);
extern void _KLTComputeGradients(void*, float, void*, void*);
extern void _KLTBulkBuildPyramidsWithGradientsULTRA(
    unsigned char**, void*, void*, void*, int, void*);
extern void _KLTCleanupUltraBuffers(void);


/* Define batch size - can be overridden at compile time */
#ifndef ULTRA_BATCH_SIZE
#define ULTRA_BATCH_SIZE 25

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

/*********************************************************************
 * Helper: Get minimum of two integers
 *********************************************************************/
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

/*********************************************************************
 * Helper: Track features using pre-computed pyramids
 *********************************************************************/
static void track_with_precomputed_pyramids(
    _KLT_Pyramid pyramid1, _KLT_Pyramid pyramid1_gradx, _KLT_Pyramid pyramid1_grady,
    _KLT_Pyramid pyramid2, _KLT_Pyramid pyramid2_gradx, _KLT_Pyramid pyramid2_grady,
    KLT_FeatureList fl,
    KLT_TrackingContext tc)
{
    float subsampling = (float) tc->subsampling;
    int ncols = pyramid1->ncols[0];
    int nrows = pyramid1->nrows[0];
    
    // For each feature, track through pyramid
    for (int indx = 0; indx < fl->nFeatures; indx++) {
        
        if (fl->feature[indx]->val < 0) continue;  // Skip lost features
        
        float xloc = fl->feature[indx]->x;
        float yloc = fl->feature[indx]->y;
        
        // Transform to coarsest resolution
        for (int r = tc->nPyramidLevels - 1; r >= 0; r--) {
            xloc /= subsampling;
            yloc /= subsampling;
        }
        
        float xlocout = xloc, ylocout = yloc;
        int val = KLT_TRACKED;
        
        // Track from coarse to fine
        for (int r = tc->nPyramidLevels - 1; r >= 0; r--) {
            xloc *= subsampling;
            yloc *= subsampling;
            xlocout *= subsampling;
            ylocout *= subsampling;
            
            val = _trackFeature(xloc, yloc, &xlocout, &ylocout,
                                pyramid1->img[r],
                                pyramid1_gradx->img[r], pyramid1_grady->img[r],
                                pyramid2->img[r],
                                pyramid2_gradx->img[r], pyramid2_grady->img[r],
                                tc->window_width, tc->window_height,
                                tc->step_factor,
                                tc->max_iterations,
                                tc->min_determinant,
                                tc->min_displacement,
                                tc->max_residue,
                                tc->lighting_insensitive);
            
            if (val == KLT_SMALL_DET || val == KLT_OOB) break;
        }
        
        // Update feature
        if (val == KLT_TRACKED && 
            !_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery)) {
            fl->feature[indx]->x = xlocout;
            fl->feature[indx]->y = ylocout;
            fl->feature[indx]->val = KLT_TRACKED;
        } else {
            fl->feature[indx]->x = -1.0;
            fl->feature[indx]->y = -1.0;
            fl->feature[indx]->val = val;
        }
    }
}

/*********************************************************************
 * MAIN - GPU-Accelerated KLT Tracking with ULTRA Mode
 *********************************************************************/
#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
    unsigned char *img_for_feature_detect;
    char fnamein[256], fnameout[256];
    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    KLT_FeatureTable ft;
    int nFeatures = N_FEATURES;
    int nFrames = 10;  // Default, will be updated
    int ncols = 0, nrows = 0;
    
    clock_t start_time, end_time;
    double cpu_time_used;
    
    // ================================================================
    // SETUP: Count frames and create tracking context
    // ================================================================
    char cmd[256];
    sprintf(cmd, "ls %simg*.pgm 2>/dev/null | wc -l", DATA_DIR);
    FILE *fp = popen(cmd, "r");
    if (fp) {
        fscanf(fp, "%d", &nFrames);
        pclose(fp);
    }
    
    if (nFrames <= 0) {
        fprintf(stderr, "❌ ERROR: No image files found in %s\n", DATA_DIR);
        return 1;
    }
    
    // Apply MAX_FRAMES limit
    if (nFrames > MAX_FRAMES) {
        printf("⚠️  Limiting to %d frames (found %d)\n", MAX_FRAMES, nFrames);
        nFrames = MAX_FRAMES;
    }
    
    printf("========================================\n");
    printf("🚀 GPU-Accelerated KLT Feature Tracking\n");
    printf("========================================\n");
    printf("Frames: %d\n", nFrames);
    printf("Features: %d\n", nFeatures);
    printf("Batch size: %d\n", ULTRA_BATCH_SIZE);
    printf("========================================\n\n");
    
    start_time = clock();
    
    tc = KLTCreateTrackingContext();
    fl = KLTCreateFeatureList(nFeatures);
    ft = KLTCreateFeatureTable(nFrames, nFeatures);
    
    tc->sequentialMode = FALSE;  // ⚠️ MUST be FALSE for ULTRA mode!
    tc->writeInternalImages = FALSE;
    tc->affineConsistencyCheck = -1;
    
    // ================================================================
    // STEP 1: Load first frame and detect features
    // ================================================================
    printf("[1/3] Loading first frame and detecting features...\n");
    sprintf(fnamein, "%simg0.pgm", DATA_DIR);
    img_for_feature_detect = pgmReadFile(fnamein, NULL, &ncols, &nrows);
    
    if (!img_for_feature_detect) {
        fprintf(stderr, "❌ ERROR: Could not load %s\n", fnamein);
        return 1;
    }
    
    printf("  Image size: %d × %d\n", ncols, nrows);
    
    KLTSelectGoodFeatures(tc, img_for_feature_detect, ncols, nrows, fl);
    KLTStoreFeatureList(fl, ft, 0);
    
    int initial_features = KLTCountRemainingFeatures(fl);
    printf("  ✅ Selected %d features\n\n", initial_features);
    
    // ================================================================
    // STEP 2: Allocate buffers for ULTRA mode
    // ================================================================
    printf("[2/3] Allocating ULTRA mode buffers...\n");
    
    KLT_PixelType **frame_buffer = malloc(ULTRA_BATCH_SIZE * sizeof(KLT_PixelType*));
    _KLT_Pyramid *pyramids = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
    _KLT_Pyramid *pyramids_gradx = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
    _KLT_Pyramid *pyramids_grady = malloc(ULTRA_BATCH_SIZE * sizeof(_KLT_Pyramid));
    
    if (!frame_buffer || !pyramids || !pyramids_gradx || !pyramids_grady) {
        fprintf(stderr, "❌ ERROR: Memory allocation failed!\n");
        return 1;
    }
    
    for (int i = 0; i < ULTRA_BATCH_SIZE; i++) {
        frame_buffer[i] = (KLT_PixelType*) malloc(ncols * nrows * sizeof(KLT_PixelType));
        pyramids[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
        pyramids_gradx[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
        pyramids_grady[i] = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
        
        if (!frame_buffer[i] || !pyramids[i] || !pyramids_gradx[i] || !pyramids_grady[i]) {
            fprintf(stderr, "❌ ERROR: Memory allocation failed at index %d!\n", i);
            return 1;
        }
    }
    
    printf("  ✅ Allocated %d frame buffers\n", ULTRA_BATCH_SIZE);
    printf("  ✅ Allocated %d pyramid sets\n\n", ULTRA_BATCH_SIZE);
    
    // ================================================================
    // STEP 3: Process frames in batches (ULTRA MODE!)
    // ================================================================
    printf("[3/3] Tracking features through %d frames...\n", nFrames - 1);
    
    double total_pyramid_time = 0.0;
    double total_tracking_time = 0.0;
    int total_batches = 0;
    
    // Process from frame 1 to nFrames-1 (frame 0 already processed)
    for (int batch_start = 1; batch_start < nFrames; batch_start += ULTRA_BATCH_SIZE) {
        
        // ============================================================
        // Determine actual batch size (handle last batch!)
        // ============================================================
        int batch_size = min(ULTRA_BATCH_SIZE, nFrames - batch_start);
        total_batches++;
        
        printf("\n  Batch %d: Frames %d-%d (%d frames)\n", 
               total_batches, batch_start, batch_start + batch_size - 1, batch_size);
        
        // ============================================================
        // Load batch of frames
        // ============================================================
        clock_t t_load = clock();
        
        for (int i = 0; i < batch_size; i++) {
            sprintf(fnamein, "%simg%d.pgm", DATA_DIR, batch_start + i);
            KLT_PixelType *loaded = pgmReadFile(fnamein, NULL, &ncols, &nrows);
            
            if (!loaded) {
                fprintf(stderr, "    ❌ ERROR: Could not load %s\n", fnamein);
                // Handle gracefully: skip this frame
                batch_size = i;  // Truncate batch
                break;
            }
            
            // Copy to buffer
            memcpy(frame_buffer[i], loaded, ncols * nrows * sizeof(KLT_PixelType));
            free(loaded);
        }
        
        if (batch_size == 0) {
            printf("    ⚠️  No frames loaded, skipping batch\n");
            continue;
        }
        
        double load_time = ((double)(clock() - t_load)) / CLOCKS_PER_SEC;
        printf("    Load: %.2f ms\n", load_time * 1000);
        
        // ============================================================
        // 🚀 ULTRA PYRAMID COMPUTATION
        // ============================================================
        clock_t t_pyramid = clock();
        
        _KLTBulkBuildPyramidsWithGradientsULTRA(
            frame_buffer,
            pyramids,
            pyramids_gradx,
            pyramids_grady,
            batch_size,
            tc);
        
        double pyramid_time = ((double)(clock() - t_pyramid)) / CLOCKS_PER_SEC;
        total_pyramid_time += pyramid_time;
        
        printf("    Pyramids: %.2f ms (%.2f ms/frame)\n", 
               pyramid_time * 1000, pyramid_time * 1000 / batch_size);
        
        // ============================================================
        // Track features through batch
        // ============================================================
        clock_t t_track = clock();
        
        for (int i = 0; i < batch_size; i++) {
            int frame_idx = batch_start + i;
            
            // For first frame in batch, need previous frame's pyramid
            _KLT_Pyramid prev_pyramid, prev_gradx, prev_grady;
            
            if (i == 0) {
                // Need to compute pyramid for frame[batch_start - 1]
                // This is a boundary case - we'll handle it simply
                if (batch_start == 1) {
                    // Previous frame is frame 0 (feature detection frame)
                    // Compute its pyramid on-the-fly (one-time cost)
                    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
                    _KLT_FloatImage floatimg = _KLTCreateFloatImage(ncols, nrows);
                    
                    _KLTToFloatImage(img_for_feature_detect, ncols, nrows, tmpimg);
                    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
                    
                    prev_pyramid = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
                    prev_gradx = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
                    prev_grady = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
                    
                    _KLTComputePyramid(floatimg, prev_pyramid, tc->pyramid_sigma_fact);
                    
                    for (int lvl = 0; lvl < tc->nPyramidLevels; lvl++) {
                        _KLTComputeGradients(prev_pyramid->img[lvl], tc->grad_sigma,
                                            prev_gradx->img[lvl], prev_grady->img[lvl]);
                    }
                    
                    _KLTFreeFloatImage(tmpimg);
                    _KLTFreeFloatImage(floatimg);
                } else {
                    // Use last pyramid from previous batch
                    prev_pyramid = pyramids[ULTRA_BATCH_SIZE - 1];
                    prev_gradx = pyramids_gradx[ULTRA_BATCH_SIZE - 1];
                    prev_grady = pyramids_grady[ULTRA_BATCH_SIZE - 1];
                }
            } else {
                // Use pyramid from previous frame in this batch
                prev_pyramid = pyramids[i - 1];
                prev_gradx = pyramids_gradx[i - 1];
                prev_grady = pyramids_grady[i - 1];
            }
            
            // Track features
            track_with_precomputed_pyramids(
                prev_pyramid, prev_gradx, prev_grady,
                pyramids[i], pyramids_gradx[i], pyramids_grady[i],
                fl, tc);
            
            // Store results
            KLTStoreFeatureList(fl, ft, frame_idx);
            
            // Optional: Replace lost features
            #ifdef REPLACE
            KLTReplaceLostFeatures(tc, frame_buffer[i], ncols, nrows, fl);
            #endif
            
            // Cleanup frame 0 pyramid if we created it
            if (i == 0 && batch_start == 1) {
                _KLTFreePyramid(prev_pyramid);
                _KLTFreePyramid(prev_gradx);
                _KLTFreePyramid(prev_grady);
            }
        }
        
        double track_time = ((double)(clock() - t_track)) / CLOCKS_PER_SEC;
        total_tracking_time += track_time;
        
        int remaining = KLTCountRemainingFeatures(fl);
        printf("    Tracking: %.2f ms (%.2f ms/frame)\n", 
               track_time * 1000, track_time * 1000 / batch_size);
        printf("    Features remaining: %d / %d\n", remaining, nFeatures);
        
        // Early exit if too few features
        if (remaining < nFeatures / 10) {
            printf("    ⚠️  Too few features remaining! Consider re-detecting.\n");
        }
    }
    
    // ================================================================
    // STEP 4: Save results and report statistics
    // ================================================================
    printf("\n========================================\n");
    printf("📊 PROCESSING COMPLETE\n");
    printf("========================================\n");
    
    printf("💾 Saving feature table...\n");
    sprintf(fnameout, "%sfeatures.txt", OUTPUT_DIR);
    KLTWriteFeatureTable(ft, fnameout, "%5.1f");
    sprintf(fnameout, "%sfeatures.ft", OUTPUT_DIR);
    KLTWriteFeatureTable(ft, fnameout, NULL);
    printf("  ✅ Saved to %s\n", OUTPUT_DIR);
    
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("\n⏱️  TIMING BREAKDOWN:\n");
    printf("  Pyramid computation: %.2f sec (%.2f ms/frame)\n",
           total_pyramid_time, total_pyramid_time * 1000 / (nFrames - 1));
    printf("  Feature tracking:    %.2f sec (%.2f ms/frame)\n",
           total_tracking_time, total_tracking_time * 1000 / (nFrames - 1));
    printf("  Other (I/O, etc):    %.2f sec\n",
           cpu_time_used - total_pyramid_time - total_tracking_time);
    printf("  ────────────────────────────────────\n");
    printf("  TOTAL:               %.2f sec\n", cpu_time_used);
    printf("  Throughput:          %.2f FPS\n\n", (nFrames - 1) / cpu_time_used);
    
    int final_features = KLTCountRemainingFeatures(fl);
    printf("📈 FEATURE STATISTICS:\n");
    printf("  Initial features:  %d\n", initial_features);
    printf("  Final features:    %d\n", final_features);
    printf("  Retention rate:    %.1f%%\n\n", 
           100.0 * final_features / initial_features);
    
    printf("✅ KLT feature tracking completed successfully!\n");
    printf("========================================\n\n");
    
    // ================================================================
    // CLEANUP
    // ================================================================
    for (int i = 0; i < ULTRA_BATCH_SIZE; i++) {
        free(frame_buffer[i]);
        _KLTFreePyramid(pyramids[i]);
        _KLTFreePyramid(pyramids_gradx[i]);
        _KLTFreePyramid(pyramids_grady[i]);
    }
    free(frame_buffer);
    free(pyramids);
    free(pyramids_gradx);
    free(pyramids_grady);
    
    _KLTCleanupUltraBuffers();  // Free GPU resources
    
    KLTFreeFeatureTable(ft);
    KLTFreeFeatureList(fl);
    KLTFreeTrackingContext(tc);
    free(img_for_feature_detect);
    
    // Print just timing for automated benchmarking
    printf("%.3f\n", cpu_time_used);
    
    return 0;
}