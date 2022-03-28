
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
    int arr1Size = env->GetArrayLength(arr1);
    int arr2Size = env->GetArrayLength(arr2);
    vector<float> jniArr1(arr1Size);
    vector<float> jniArr2(arr2Size);

    base::JavaFloatArrayToFloatVector(env, arr1, &jniArr1);
    base::JavaFloatArrayToFloatVector(env, arr2, &jniArr2);

    float out;
    for(int i = 0; i < arr1Size; i++){
        auto temp = jniArr1[i] * jniArr2[i];
        out += temp;
    }

    return out;
}

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_jiwon_neon_1simd_Operations_dotNeon(JNIEnv *env, jobject thiz,
                                             jfloatArray arr1,
                                             jfloatArray arr2) {

    int arr1Size = env->GetArrayLength(arr1);
    int arr2Size = env->GetArrayLength(arr2);
    vector<float> jniArr1(arr1Size);
    vector<float> jniArr2(arr2Size);

    base::JavaFloatArrayToFloatVector(env, arr1, &jniArr1);
    base::JavaFloatArrayToFloatVector(env, arr2, &jniArr2);

    auto start = std::chrono::system_clock::now();

    short transferSize = 4;
    short segments = arr1Size / transferSize;

    // 4-element vector of zeros
    float32x4_t partialSums = vdupq_n_f32(0);

    for(int i = 0 ; i < segments; i++){
        short offset = i * transferSize;
        float32x4_t a1 = vld1q_f32(jniArr1.data() + offset);
        float32x4_t a2 = vld1q_f32(jniArr2.data() + offset);
        partialSums = vmlaq_f32(partialSums, a1, a2);
    }

    float partialSum[arr1Size];
    vst1q_f32(partialSum, partialSums);

    float result = 0;
    for(int i = 0; i < transferSize; i++){
        result += partialSum[i];
    }

    return result;

}