// Native Objective-C++ CoreML Apple Neural Engine (ANE) Runner for Apple Silicon.
// Called via ctypes.CDLL (which automatically releases the Python GIL!) so CoreML ANE
// executes 100% concurrently with Apple M5 Metal GPU (MLX) and CPU (vDSP/NEON) with
// zero GIL contention and zero Python dictionary / array copy overhead.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct CoreMLRunner {
    MLModel *model;
    NSString *inputName;
    NSString *outputName;
    int32_t dim0;
    int32_t dim1;
    int32_t dim2;
    MLMultiArray *inputArr;
    float *inputPtr;
};

extern "C" {

// compute_units: 0 = CPU_ONLY, 1 = CPU_AND_GPU, 2 = ALL, 3 = CPU_AND_NE
void* coreml_runner_create(
    const char *pkg_path,
    const char *input_name,
    const char *output_name,
    int32_t d0,
    int32_t d1,
    int32_t d2,
    int32_t compute_units
) {
    @autoreleasepool {
        NSError *err = nil;
        NSString *pathStr = [NSString stringWithUTF8String:pkg_path];
        NSURL *srcUrl = [NSURL fileURLWithPath:pathStr];

        // Compile .mlpackage to .mlmodelc (cached next to package if possible)
        NSString *compiledPath = [[pathStr stringByDeletingPathExtension] stringByAppendingString:@".mlmodelc"];
        NSURL *compiledUrl = [NSURL fileURLWithPath:compiledPath];
        NSFileManager *fm = [NSFileManager defaultManager];
        if (![fm fileExistsAtPath:compiledPath]) {
            NSURL *tmpCompiled = [MLModel compileModelAtURL:srcUrl error:&err];
            if (err || !tmpCompiled) {
                fprintf(stderr, "[coreml_ane] compileModelAtURL failed: %s\n",
                        err ? [[err localizedDescription] UTF8String] : "unknown");
                return nullptr;
            }
            [fm copyItemAtURL:tmpCompiled toURL:compiledUrl error:nil];
            if (![fm fileExistsAtPath:compiledPath]) {
                compiledUrl = tmpCompiled;
            }
        }

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        if (compute_units == 0) {
            cfg.computeUnits = MLComputeUnitsCPUOnly;
        } else if (compute_units == 1) {
            cfg.computeUnits = MLComputeUnitsCPUAndGPU;
        } else if (compute_units == 2) {
            cfg.computeUnits = MLComputeUnitsAll;
        } else {
            cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        }

        MLModel *model = [MLModel modelWithContentsOfURL:compiledUrl configuration:cfg error:&err];
        if (err || !model) {
            fprintf(stderr, "[coreml_ane] modelWithContentsOfURL failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return nullptr;
        }

        NSArray<NSNumber *> *shape = @[@(d0), @(d1), @(d2)];
        MLMultiArray *inArr = [[MLMultiArray alloc] initWithShape:shape
                                                         dataType:MLMultiArrayDataTypeFloat32
                                                            error:&err];
        if (err || !inArr) {
            fprintf(stderr, "[coreml_ane] MLMultiArray alloc failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return nullptr;
        }

        CoreMLRunner *r = new CoreMLRunner();
        r->model = model;
        r->inputName = [NSString stringWithUTF8String:input_name];
        r->outputName = [NSString stringWithUTF8String:output_name];
        r->dim0 = d0;
        r->dim1 = d1;
        r->dim2 = d2;
        r->inputArr = inArr;
        r->inputPtr = (float *)inArr.dataPointer;
        return (void *)r;
    }
}

// Runs inference synchronously on ANE/GPU (with Python GIL released by ctypes!).
// Copies `in_count` float32s from `in_data` into reusable MLMultiArray, predicts,
// and copies up to `out_max` float32s into `out_data`. Returns number of output floats written, or <0 on error.
int32_t coreml_runner_predict(
    void *handle,
    const float *in_data,
    int32_t in_count,
    float *out_data,
    int32_t out_max
) {
    if (!handle) return -1;
    CoreMLRunner *r = (CoreMLRunner *)handle;
    @autoreleasepool {
        memcpy(r->inputPtr, in_data, (size_t)in_count * sizeof(float));
        NSError *err = nil;
        MLDictionaryFeatureProvider *inFeat = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{r->inputName: [MLFeatureValue featureValueWithMultiArray:r->inputArr]}
                         error:&err];
        if (err || !inFeat) return -2;

        id<MLFeatureProvider> outFeat = [r->model predictionFromFeatures:inFeat error:&err];
        if (err || !outFeat) {
            fprintf(stderr, "[coreml_ane] predict failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return -3;
        }

        MLFeatureValue *fv = [outFeat featureValueForName:r->outputName];
        if (!fv || fv.type != MLFeatureTypeMultiArray) return -4;
        MLMultiArray *outArr = fv.multiArrayValue;
        int32_t count = (int32_t)outArr.count;
        int32_t n = count < out_max ? count : out_max;

        if (outArr.dataType == MLMultiArrayDataTypeFloat32) {
            memcpy(out_data, outArr.dataPointer, (size_t)n * sizeof(float));
        } else if (outArr.dataType == MLMultiArrayDataTypeFloat16) {
            const __fp16 *src = (const __fp16 *)outArr.dataPointer;
            for (int32_t i = 0; i < n; ++i) {
                out_data[i] = (float)src[i];
            }
        } else {
            const double *src = (const double *)outArr.dataPointer;
            for (int32_t i = 0; i < n; ++i) {
                out_data[i] = (float)src[i];
            }
        }
        return n;
    }
}

void coreml_runner_destroy(void *handle) {
    if (!handle) return;
    @autoreleasepool {
        CoreMLRunner *r = (CoreMLRunner *)handle;
        r->model = nil;
        r->inputName = nil;
        r->outputName = nil;
        r->inputArr = nil;
        delete r;
    }
}

} // extern "C"
