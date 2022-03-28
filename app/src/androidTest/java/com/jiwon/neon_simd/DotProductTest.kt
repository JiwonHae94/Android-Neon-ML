package com.jiwon.neon_simd

import com.jiwon.neon_simd.helper.TestHelper
import com.jiwon.neon_simd.helper.TestHelper.generateRandomFloatArray
import org.junit.Test
import kotlin.math.roundToInt
import kotlin.random.Random

class DotProductTest{
    private val NumTrial = 1000

    @Test
    fun testNativDotProduct(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0

        // neon improves performance by .58
        for(no in 0 until NumTrial){
            val (arr1, arr2) = generateRandomFloatArray(512)
            val startTimeNative = System.currentTimeMillis()
            val rsltNative = arr1.zip(arr2).sumOf { pair ->
                (pair.first * pair.second).toDouble()
            }

            println("dot product : ${rsltNative}")
            val endTimeNative = System.currentTimeMillis()
            nativeTimeTaken += endTimeNative - startTimeNative

            val startTimeCPP = System.currentTimeMillis()
            val rsltCPP = Operations.dotJNI(arr1, arr2).toDouble()

            println("dot product cpp : ${rsltCPP}")
            val endTimeCPP = System.currentTimeMillis()
            cppTimeTaken += endTimeCPP - startTimeCPP

            val startTimeNeon = System.currentTimeMillis()
            val rsltNeon = Operations.dotNeon(arr1, arr2).toDouble()
            println("dot product neon: ${rsltNeon}")
            val endTimeNeon = System.currentTimeMillis()
            neonTimeTaken += endTimeNeon - startTimeNeon

            println("native : ${rsltNative}")
            println("cpp : ${rsltCPP}")
            println("neon : ${rsltNeon}")
            
            val result = TestHelper.compareResults(rsltNative, rsltCPP, rsltNeon)
            assert(result)
        }

        println("time taken native : ${nativeTimeTaken / NumTrial}")
        println("time taken c++ : ${cppTimeTaken / NumTrial}")
        println("time taken neon : ${neonTimeTaken / NumTrial}")
    }
}