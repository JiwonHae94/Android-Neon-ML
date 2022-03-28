# Neon - ARM Advanced SIMD in AOS
This project demonstrates the use of Neon in AOS and performance improvement obtained by the implementation

## Neon
### What is Neon
[`Neon`](https://developer.arm.com/documentation/102467/0100/What-is-Neon-?lang=en) is the implementation of Arm's Advanced SIMD architecture. 
Neon provides scalar/vector instruction and registered (shared with the FPU) comparable to MMX/SSE/SdNow! in the x86World

### Purpose of Neon
* Thirty-two 128-bit vector registers, each capable of containing multiple lanes of data.
* SIMD instructions to operate simultaneously on those multiple lanes of data.

### Neon Intrinsics
`Neon Intrinsics` are functions whose precise implementation is known to a compiler. 
The Neon intrinsics are a set of C and C++ functions defined in arm_neon.h which are supported by the Arm compilers and GCC

### Compatiability
- Some of ARMv7
- All ARMv8

## Set up
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

implement matrix / multi-array / vector operation through neon 
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
Operation | Num Trials | Native(Kotlin) | C++ | Neon |
--- | --- | --- | --- |--- |
Dot Product | 1000 | 0.119ms | 0.076ms | 0.069ms |

_some of the functions used in this project can be found [here](https://developer.arm.com/documentation/102467/0100/Matrix-multiplication-example?lang=en)_

credits to _neon developers and arm developers for providing detailed guidance_