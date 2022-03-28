//
// Created by Jiwon_Hae on 2022/03/02.
//

#include <iostream>
#include <android/log.h>
#include <vector>
#include "jni.h"

namespace base {
    void JavaIntArrayToIntVector(JNIEnv *env, jintArray arr, std::vector<int> *out){
        int len = env->GetArrayLength(arr);
        if(!len)
            return;
        env->GetIntArrayRegion(arr, 0, len, out->data());
    }

    void JavaFloatArrayToFloatVector(JNIEnv *env, jfloatArray arr, std::vector<float> *out){
        int len = env->GetArrayLength(arr);
        if(!len)
            return;
        env->GetFloatArrayRegion(arr, 0, len, out->data());
    }

    void JavaDoubleArrayToDoubleVector(JNIEnv *env, jdoubleArray arr, std::vector<double> *out){
        int len = env->GetArrayLength(arr);
        if(!len)
            return;
        env->GetDoubleArrayRegion(arr, 0, len, out->data());
    }

    void JavaLongArrayToLongVector(JNIEnv *env, jlongArray arr, std::vector<long> *out){
        int len = env->GetArrayLength(arr);
        if(!len)
            return;
        env->GetLongArrayRegion(arr, 0, len, out->data());
    }
}