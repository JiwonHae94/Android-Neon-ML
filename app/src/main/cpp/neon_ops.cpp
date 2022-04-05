
//
// Created by Jiwon_Hae on 2022/03/28.
//
#include <jni.h>
#include <vector>
#include <arm_neon.h>
#include <neon_math.hpp>
#include <jni_array.hpp>
#include "string"
#include "chrono"
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>

#define LOG_TAG "NEON_SIMD"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG , LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO , LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN , LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR , LOG_TAG, __VA_ARGS__)

#if defined(HAVE_NEON) && defined(HAVE_NEON_X86)
/*
  * The latest version and instruction for NEON_2_SSE.h is at:
  *    https://github.com/intel/ARM_NEON_2_x86_SSE
  */
  #include "NEON_2_SSE.h"
#elif defined(HAVE_NEON)
    #include <arm_neon.h>
#endif

using namespace std;


double msElapsedTime(std::chrono::system_clock::time_point start) {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::chrono::system_clock::time_point now() {
    return std::chrono::system_clock::now();
}


extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_dotJNI(JNIEnv *env, jobject thiz,
                                            jfloatArray arr1,
                                            jfloatArray arr2) {
    int len = env->GetArrayLength(arr1);
    float* jniArr1 = env->GetFloatArrayElements(arr1, 0);
    float* jniArr2 = env->GetFloatArrayElements(arr2, 0);

    float out;
    for(int i = 0; i < len; i++){
        auto temp = jniArr1[i] * jniArr2[i];
        out += temp;
    }

    env->ReleaseFloatArrayElements(arr1, jniArr1, 0);
    env->ReleaseFloatArrayElements(arr2, jniArr2, 0);

    return out;
}

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_dotNeon(JNIEnv *env, jobject thiz,
                                             jfloatArray arr1,
                                             jfloatArray arr2) {

    int len = env->GetArrayLength(arr1);
    float* jniArr1 = env->GetFloatArrayElements(arr1, 0);
    float* jniArr2 = env->GetFloatArrayElements(arr2, 0);

    short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    float32x4_t partialSums = vdupq_n_f32(0);

    for(int i = 0 ; i < segments; i++){
        short offset = i * transferSize;
        float32x4_t a1 = vld1q_f32(jniArr1 + offset);
        float32x4_t a2 = vld1q_f32(jniArr2 + offset);
        partialSums = vmlaq_f32(partialSums, a1, a2);
    }

    env->ReleaseFloatArrayElements(arr1, jniArr1, 0);
    env->ReleaseFloatArrayElements(arr2, jniArr2, 0);

    float partialSum[len];
    vst1q_f32(partialSum, partialSums);

    float result = 0;
    for(int i = 0; i < transferSize; i++){
        result += partialSum[i];
    }

    return result;

}
extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_cosineSimilarityCPP(JNIEnv *env, jobject thiz,
                                                         jfloatArray arr1, jfloatArray arr2) {

    int len = env->GetArrayLength(arr1);
    float* jniArr1 = env->GetFloatArrayElements(arr1, 0);
    float* jniArr2 = env->GetFloatArrayElements(arr2, 0);

    float sumProduct = 0, sumSqA = 0, sumSqB = 0;

    for(int i = 0; i < len; i++){
        sumProduct += jniArr1[i] * jniArr2[i];
        sumSqA += pow(jniArr1[i], 2.0);
        sumSqB += pow(jniArr2[i], 2.0);
    }

    env->ReleaseFloatArrayElements(arr1, jniArr1, 0);
    env->ReleaseFloatArrayElements(arr2, jniArr2, 0);

    return sumProduct / (sqrt(sumSqA) * sqrt(sumSqB));
}

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_cosineSimilarityNeon(JNIEnv *env, jobject thiz,
                                                          jfloatArray arr1, jfloatArray arr2) {
    int len = env->GetArrayLength(arr1);
    short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    float32x4_t partialSums = vdupq_n_f32(0);
    float32x4_t partialSqAs = vdupq_n_f32(0);
    float32x4_t partialSqBs = vdupq_n_f32(0);

    float* jniArr1 = env->GetFloatArrayElements(arr1, 0);
    float* jniArr2 = env->GetFloatArrayElements(arr2, 0);

    for(int i = 0 ; i < segments; i++){
        short offset = i * transferSize;
        float32x4_t a1 = vld1q_f32(jniArr1 + offset);
        float32x4_t a2 = vld1q_f32(jniArr2 + offset);

        partialSums = vmlaq_f32(partialSums, a1, a2);
        partialSqAs = vmlaq_f32(partialSqAs, a1, a1);
        partialSqBs = vmlaq_f32(partialSqBs, a2, a2);
    }

    env->ReleaseFloatArrayElements(arr1, jniArr1, 0);
    env->ReleaseFloatArrayElements(arr2, jniArr2, 0);

    float partialSum[len];
    float partialSqA[len];
    float partialSqB[len];
    vst1q_f32(partialSum, partialSums);
    vst1q_f32(partialSqA, partialSqAs);
    vst1q_f32(partialSqB, partialSqBs);

    float result = 0;
    float sumSqA = 0;
    float sumSqB = 0;
    for(int i = 0; i < transferSize; i++){
        result += partialSum[i];
        sumSqA += partialSqA[i];
        sumSqB += partialSqB[i];
    }

    return result / (sqrt(sumSqA) * sqrt(sumSqB));

}


extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_jiwon_neon_1simd_Operations_softmaxJNI(JNIEnv *env, jobject thiz, jfloatArray arr1) {
    int len = env->GetArrayLength(arr1);
    float* jniArr = env->GetFloatArrayElements(arr1, 0);

    float _max = *std::max_element(jniArr, jniArr + len);
    float exp_x[len];
    float sum = 0;

    for(int i = 0 ; i < len; i++){
        exp_x[i] = exp(jniArr[i] - _max);
        sum += exp_x[i];
    }

    for(int i = 0; i < len; i++){
        exp_x[i] /= sum;
    }

    env->ReleaseFloatArrayElements(arr1, jniArr, 0);
    jfloatArray outArray =env->NewFloatArray(len);
    env->SetFloatArrayRegion(outArray, 0, len, exp_x);
    return outArray;
}
extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_jiwon_neon_1simd_Operations_softmaxNeon(JNIEnv *env, jobject thiz, jfloatArray arr1) {
    // retrieve length and elements from arr
    int len = env->GetArrayLength(arr1);
    float* jniArr = env->GetFloatArrayElements(arr1, 0);

    short transferSize = 4;
    short numSegments = len / 4;

    // partial max to get max value
    float32x4_t partialMax = vdupq_n_f32(0);
    for(int i = 0; i < numSegments; i++){
        short offset = i * transferSize;

        // get max of vectors
        float32x4_t a1 = vld1q_f32(jniArr + offset);
        partialMax = vmaxq_f32(partialMax, a1);
    }

    float maxArray[4];
    vst1q_f32(maxArray, partialMax);

    // max obtained
    float maxValue = *max_element(maxArray, maxArray + transferSize);

    // create neon with _max obtained
    float32x4_t max = vmovq_n_f32(maxValue);

    float exp_x[len];
    float32x4_t partialSums = vdupq_n_f32(0);

    for(int i = 0; i < numSegments; i++){
        short offset = i * transferSize;

        // obtain sublist
        float32x4_t a1 = vld1q_f32(jniArr + offset);

        // exp
        a1 = exp_ps(vsubq_f32(a1, max));

        vst1q_f32(exp_x + offset, a1);

        // accumulate partial sums
        partialSums = vaddq_f32(partialSums, a1);
    }

    // release jni arr as it's no longer in use
    env->ReleaseFloatArrayElements(arr1, jniArr, 0);

    // retrieve sum of the input arr
    float partialSum[transferSize];
    vst1q_f32(partialSum, partialSums);

    float sum = 0;
    for(int i = 0; i < 4; i++){
        sum += partialSum[i];
    }

    // convert mul -> div
    sum = 1 / sum;

    for(int i = 0; i < numSegments; i++){
        short offset = i * transferSize;

        // get sublist
        float32x4_t a1 = vld1q_f32(exp_x + offset);

        // div sublist by sum : a1 /= sum
        vst1q_f32(exp_x + offset, vmulq_n_f32(a1, sum));
    }

    // convert to native array
    jfloatArray outArray =env->NewFloatArray(len);
    env->SetFloatArrayRegion(outArray, 0, len, exp_x);
    return outArray;;
}

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_sumNeon(JNIEnv *env, jobject thiz, jfloatArray arr1) {
    float* jarr = env->GetFloatArrayElements(arr1, 0);
    int len = env->GetArrayLength(arr1);

    int dim4 = len / 4;  //Array length divided by 4 integer
    int left4 = len - dim4 * 4;  //Array length divided by 4 remainder
    float32x4_t sum_vec =  vdupq_n_f32 ( 0.0 ) ; //Define the register used to temporarily store the accumulation result and initialize it Is 0

    short offset = 0;

    for(int i = 0; i < dim4; i++){
        offset = i * 4;
        float32x4_t data_vec =  vld1q_f32(jarr + offset);
        sum_vec =  vaddq_f32 (sum_vec , data_vec);
    }

    float sum = vgetq_lane_f32(sum_vec,  0) + vgetq_lane_f32 ( sum_vec ,  1 ) + vgetq_lane_f32 ( sum_vec ,  2 ) + vgetq_lane_f32 ( sum_vec ,  3 ) ; //Add all the elements in the accumulation result register to get the final accumulated value
    for(int j = 0; j < left4; j++){
        sum += jarr[j + offset + 4];
    }

    env->ReleaseFloatArrayElements(arr1, jarr, 0);
    return sum;

}



extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_sumJNI(JNIEnv *env, jobject thiz, jfloatArray arr1) {
    float* jarr = env->GetFloatArrayElements(arr1, 0);
    float sum = 0.0;
    for(int i = 0 ; i < env->GetArrayLength(arr1); i++){
        sum += jarr[i];
    }
    env->ReleaseFloatArrayElements(arr1, jarr, 0);
    return sum;
}
extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_averageJNI(JNIEnv *env, jobject thiz, jfloatArray arr1) {
    float* jarr = env->GetFloatArrayElements(arr1, 0);
    float sum = 0.0;
    int len = env->GetArrayLength(arr1);
    for(int i = 0 ; i < len; i++){
        sum += jarr[i];
    }

    //env->ReleaseFloatArrayElements(arr1, jarr, 0);
    return sum / len;
}

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_averageNeon(JNIEnv *env, jobject thiz, jfloatArray arr1) {
    float* jarr = env->GetFloatArrayElements(arr1, 0);
    int len = env->GetArrayLength(arr1);
    int dim4 = len >> 2;  //Array length divided by 4 integer
    int left4 = len & 3; // len - dim4 * 4 ;  //Array length divided by 4 remainder
    float32x4_t sum_vec =  vdupq_n_f32 ( 0.0 ) ; //Define the register used to temporarily store the accumulation result and initialize it Is 0

    short offset = 0;

    for(int i = 0; i < dim4; i++){
        offset = i * 4;
        float32x4_t data_vec =  vld1q_f32(jarr + offset);
        sum_vec =  vaddq_f32 (sum_vec , data_vec);
    }

    float sum = vgetq_lane_f32(sum_vec,  0) + vgetq_lane_f32 ( sum_vec ,  1 ) + vgetq_lane_f32 ( sum_vec ,  2 ) + vgetq_lane_f32 ( sum_vec ,  3 ) ; //Add all the elements in the accumulation result register to get the final accumulated value
    for(int j = 0; j < left4; j++){
        sum += jarr[j + offset + 4];
    }

    //env->ReleaseFloatArrayElements(arr1, jarr, 0);
    return sum / len;
}