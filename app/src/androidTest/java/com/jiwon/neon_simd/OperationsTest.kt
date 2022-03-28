package com.jiwon.neon_simd

import org.junit.Test
import kotlin.math.roundToInt
import kotlin.random.Random

class OperationsTest{
    private val NumTrial = 1000
    private fun generateRandomDoubleInput(len : Int) : Pair<DoubleArray, DoubleArray>{
        val arr1 = Array(len){ Random.nextDouble() }
        val arr2 = Array(len){ Random.nextDouble() }
        return Pair(arr1.toDoubleArray(), arr2.toDoubleArray())
    }

    private fun generateRandomFloatInput(len : Int) : Pair<FloatArray, FloatArray>{
        val arr1 = Array(len){ Random.nextFloat() }
        val arr2 = Array(len){ Random.nextFloat() }
        return Pair(arr1.toFloatArray(), arr2.toFloatArray())
    }

    @Test
    fun testNativDotProduct(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0

        // neon improves performance by .58
        for(no in 0 until NumTrial){
            val (arr1, arr2) = generateRandomFloatInput(512)
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

            assert((rsltNative.roundToInt() == rsltCPP.roundToInt()) && (rsltCPP.roundToInt() == rsltNeon.roundToInt()))
        }

        println("time taken native : ${nativeTimeTaken / NumTrial}")
        println("time taken c++ : ${cppTimeTaken / NumTrial}")
        println("time taken neon : ${neonTimeTaken / NumTrial}")
    }
}