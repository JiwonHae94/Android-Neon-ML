//
// Created by Jiwon_Hae on 2022/03/28.
//
#include "jni.h"
#include "string"
#include "arm_neon.h"
#include "chrono"
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>

using namespace std;

short* generateRamp(short startValue, short len) {
    short* ramp = new short[len];
    for(short i = 0; i < len; i++) {
        ramp[i] = startValue + i;
    }
    return ramp;
}

double msElapsedTime(std::chrono::system_clock::time_point start) {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::chrono::system_clock::time_point now() {
    return std::chrono::system_clock::now();
}



float dot(const std::vector<float>& a, const std::vector<float>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

float norm(const std::vector<float>& a) {
    return std::sqrt(dot(a, a));
}

float cos(const std::vector<float>& a, const std::vector<float>& b) {
    return dot(a, b) / (norm(a) * norm(b));
}

float dotNeon(const std::vector<float>& a, const std::vector<float>& b) {
    auto a_data = a.data();
    auto b_data = b.data();

    float32x4_t x, y, z = vmovq_n_f32(0.0);

    for (int i = 0; i < 128; i += 4) {
        x = vld1q_f32(&a_data[i]);
        y = vld1q_f32(&b_data[i]);
        z = vfmaq_f32(z, x, y);
    }

    return vaddvq_f32(z);
}



int dotProduct(short* vector1, short* vector2, short len) {
    int result = 0;

    for(short i = 0; i < len; i++) {
        result += vector1[i] * vector2[i];
    }
    return result;
}


int dotProductNeon(short* vector1, short* vector2, short len) {
    const short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    int32x4_t partialSumsNeon = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments)
    for(short i = 0; i < segments; i++) {
        // Load vector elements to registers
        short offset = i * transferSize;
        int16x4_t vector1Neon = vld1_s16(vector1 + offset);
        int16x4_t vector2Neon = vld1_s16(vector2 + offset);

        // Multiply and accumulate: partialSumsNeon += vector1Neon * vector2Neon
        partialSumsNeon = vmlal_s16(partialSumsNeon, vector1Neon, vector2Neon);
    }

    // Store partial sums
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);

    // Sum up partial sums
    int result = 0;
    for(short i = 0; i < transferSize; i++) {
        result += partialSums[i];
    }

    return result;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_jiwon_neon_1simd_MainActivity_dotProduct(JNIEnv *env, jobject thiz) {

    // Ramp length and number of trials
    const int rampLength = 1024;
    const int trials = 10000;

    // Generate two input vectors
    // (0, 1, ..., rampLength - 1)
    // (100, 101, ..., 100 + rampLength-1)
    auto ramp1 = generateRamp(0, rampLength);
    auto ramp2 = generateRamp(100, rampLength);

    // Without NEON intrinsics
    // Invoke dotProduct and measure performance
    int lastResult = 0;

    auto start = now();
    for(int i = 0; i < trials; i++) {
        lastResult = dotProduct(ramp1, ramp2, rampLength);
    }

    auto elapsedTime = msElapsedTime(start);

    // With NEON intrinsics
    // Invoke dotProductNeon and measure performance
    int lastResultNeon = 0;

    start = now();
    for(int i = 0; i < trials; i++) {
        lastResultNeon = dotProductNeon(ramp1, ramp2, rampLength);
    }
    auto elapsedTimeNeon = msElapsedTime(start);

    // Clean up
    delete ramp1, ramp2;

    // Display results
    std::string resultsString =
            "----==== NO NEON ====----\nResult: " + to_string(lastResult)
            + "\nElapsed time: " + to_string((int)elapsedTime) + " ms"
            + "\n\n----==== NEON ====----\n"
            + "Result: " + to_string(lastResultNeon)
            + "\nElapsed time: " + to_string((int)elapsedTimeNeon) + " ms";

    return env->NewStringUTF(resultsString.c_str());
}