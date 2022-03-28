
//
// Created by Jiwon_Hae on 2022/03/28.
//
#include <jni.h>
#include <vector>
#include <arm_neon.h>
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