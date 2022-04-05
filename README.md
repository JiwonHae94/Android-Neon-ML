# Neon - ARM Advanced SIMD in AOS
This project demonstrates the use of Neon in AOS and performance improvement obtained by the implementation

## Neon
### What is Neon
[`Neon`](https://developer.arm.com/documentation/102467/0100/What-is-Neon-?lang=en) is the implementation of Arm's Advanced SIMD architecture. 
Neon provides scalar/vector instruction and registered (shared with the FPU) comparable to MMX/SSE/SdNow! in the x86World

### Purpose of Neon
* 4 Thirty-two 128-bit vector registers, each capable of containing multiple lanes of data.
* SIMD instructions to operate simultaneously on those multiple lanes of data.

### Neon Intrinsics
`Neon Intrinsics` are functions whose precise implementation is known to a compiler. 
The Neon intrinsics are a set of C and C++ functions defined in arm_neon.h which are supported by the Arm compilers and GCC

### Compatiability
- Some of ARMv7
- All ARMv8

### Instructions
- Instructions implementable can be found [here](https://developer.arm.com/architectures/instruction-sets/intrinsics)

### Syntax
Enable ARM-NEON on application's gradle script's defaultConfig section
```gradle
  defaultConfig {
        ...
        externalNativeBuild {
            cmake {
                cppFlags ''
                arguments "-DANDROID_ARM_NEON=ON"
            }
        }

        ndk{
            abiFilters "armeabi-v7a", "arm64-v8a", "x86"
        }
        ...
    }


    externalNativeBuild {
        cmake {
            path file('src/main/cpp/CMakeLists.txt')
            version '3.18.1'
        }
    }

```
include `arm_neon.h` in the targeted cpp files
```C++
...
#include "arm_neon.h"
...
```

implement operations with neon instructions
```C++

float dotProductNeon(float* vector1, float* vector2, short len) {
    const short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    float32x4_t partialSums = vdupq_n_f32(0);

    // Main loop (note that loop index goes through segments)
    for(int i = 0 ; i < segments; i++){
        short offset = i * transferSize;
        float32x4_t a1 = vld1q_f32(vector1.data() + offset);
        float32x4_t a2 = vld1q_f32(vector1.data() + offset);
        partialSums = vmlaq_f32(partialSums, a1, a2);
    }
  
    // store partial values
    float partialSum[arr1Size];
    vst1q_f32(partialSum, partialSums);

    // Sum up partial sums
    float result = 0;
    for(int i = 0; i < transferSize; i++){
        result += partialSum[i];
    }

    return result;
}
```

## Performance Table
Operation | Test Device | Input Type | Input Dims | Num Trials | Native(Kotlin) (ms) | C++ (ms) | Neon (ms) |
--- | --- |--- | --- | --- |--- | --- |--- |
Dot Product | LG-G7 | FloatArray | 1 x 512 |10000 | 0.0758 | 0.0066 | 0.0048 |
Cosine Similarity | LG-G7 | FloatArray| 1 x 512 |10000 | 0.0303 | 0.0506 | 0.01 |
Softmax 1D | LG-G7 | FloatArray |1 x 512 |10000 |  0.0998 | 0.0257 | 0.0597 |
Average | LG-G7 | FloatArray |1 x 512 |10000 |  0.0011 | 0.0067 | 0.0031 |


_some of the functions used in this project can be found [here](https://developer.arm.com/documentation/102467/0100/Matrix-multiplication-example?lang=en)_

credits to _neon developers and arm developers for providing detailed guidance_
