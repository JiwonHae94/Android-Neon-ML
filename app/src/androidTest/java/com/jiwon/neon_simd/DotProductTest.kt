package com.jiwon.neon_simd

import com.jiwon.neon_simd.helper.TestHelper
import com.jiwon.neon_simd.helper.TestHelper.generateRandomFloatArray
import org.junit.Test

class DotProductTest{
    // our gallery size
    private val NumTrial = 10000

    @Test
    fun testNativDotProduct(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0
        val opTag = "dot product"

        // neon improves performance by .58
        for(no in 0 until NumTrial){
            val (arr1, arr2) = generateRandomFloatArray(512)
            val startTimeNative = System.currentTimeMillis()
            val rsltNative = Operations.dotMatrix(arr1, arr2)
            val endTimeNative = System.currentTimeMillis()
            nativeTimeTaken += endTimeNative - startTimeNative

            val startTimeCPP = System.currentTimeMillis()
            val rsltCPP = Operations.dotJNI(arr1, arr2)
            val endTimeCPP = System.currentTimeMillis()
            cppTimeTaken += endTimeCPP - startTimeCPP

            val startTimeNeon = System.currentTimeMillis()
            val rsltNeon = Operations.dotNeon(arr1, arr2)
            val endTimeNeon = System.currentTimeMillis()
            neonTimeTaken += endTimeNeon - startTimeNeon

            val result = TestHelper.compareResults(rsltNative, rsltCPP, rsltNeon)
            assert(result)
        }

        println("$opTag time taken native : ${nativeTimeTaken / NumTrial}")
        println("$opTag time taken c++ : ${cppTimeTaken / NumTrial}")
        println("$opTag time taken neon : ${neonTimeTaken / NumTrial}")
    }

    @Test
    fun testCosineSimilarity(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0
        val opTag = "cosine similarity"

        // neon improves performance by .58
        for(no in 0 until NumTrial){
            val (arr1, arr2) = generateRandomFloatArray(512)

            val startTimeNative = System.currentTimeMillis()
            val rsltNative = Operations.cosineSimilarity(arr1.clone(), arr2.clone())
            val endTimeNative = System.currentTimeMillis()
            nativeTimeTaken += endTimeNative - startTimeNative

            val startTimeCPP = System.currentTimeMillis()
            val rsltCPP = Operations.cosineSimilarityCPP(arr1.clone(), arr2.clone())
            val endTimeCPP = System.currentTimeMillis()
            cppTimeTaken += endTimeCPP - startTimeCPP

            val startTimeNeon = System.currentTimeMillis()
            val rsltNeon = Operations.cosineSimilarityNeon(arr1.clone(), arr2.clone())
            val endTimeNeon = System.currentTimeMillis()

            neonTimeTaken += endTimeNeon - startTimeNeon
            val result = TestHelper.compareResults(rsltNative, rsltCPP, rsltNeon)
            assert(result)
        }

        println("$opTag time taken native : ${nativeTimeTaken / NumTrial}")
        println("$opTag time taken c++ : ${cppTimeTaken / NumTrial}")
        println("$opTag time taken neon : ${neonTimeTaken / NumTrial}")
    }
}