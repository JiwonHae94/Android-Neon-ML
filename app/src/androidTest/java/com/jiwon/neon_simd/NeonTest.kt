package com.jiwon.neon_simd

import com.jiwon.neon_simd.helper.TestHelper
import com.jiwon.neon_simd.helper.TestHelper.generateRandomFloatArray
import org.junit.Test
import java.text.DecimalFormat
import kotlin.random.Random

class NeonTest{
    // our gallery size
    private val NumTrial = 10000

    @Test
    fun testDotProduct(){
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

            val result = TestHelper.compareResults(rsltNative, rsltCPP, rsltNeon, marginOfDifference = 0.03f)
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

    @Test
    fun testSoftmax1D(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0
        val opTag = "softmax"

        // neon improves performance by .58
        for(no in 0 until NumTrial){
            val (arr1, arr2) = generateRandomFloatArray(512 * 2)

            val startTimeNative = System.currentTimeMillis()
            val rsltNative = Operations.softmax(arr1)
            val endTimeNative = System.currentTimeMillis()
            nativeTimeTaken += endTimeNative - startTimeNative

            val startTimeCPP = System.currentTimeMillis()
            val rsltCPP = Operations.softmaxJNI(arr1)
            val endTimeCPP = System.currentTimeMillis()
            cppTimeTaken += endTimeCPP - startTimeCPP

            val startTimeNeon = System.currentTimeMillis()
            val rsltNeon = Operations.softmaxNeon(arr1)
            val endTimeNeon = System.currentTimeMillis()
            neonTimeTaken += endTimeNeon - startTimeNeon

            val result = TestHelper.compareResults(rsltNative.maxOrNull()!!, rsltCPP.maxOrNull()!!, rsltNeon.maxOrNull()!!)
            assert(result)
        }

        println("$opTag time taken native : ${nativeTimeTaken / NumTrial}")
        println("$opTag time taken c++ : ${cppTimeTaken / NumTrial}")
        println("$opTag time taken neon : ${neonTimeTaken / NumTrial}")
    }

    @Test
    fun testSoftmaxNeon(){
        val (arr1, arr2) = generateRandomFloatArray(512)
        Operations.softmaxNeon(arr1)
    }

    @Test
    fun testSumElapsedTime(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0
        val opTag = "sum"
        val numTrial = 100

        // neon improves performance by .58
        for(no in 0 until numTrial){
            val arrSize = Random.nextInt(300, 5000)
            val arr1 = FloatArray(arrSize){ Random.nextFloat() }

            val startTimeNative = System.currentTimeMillis()
            val rsltNative = Operations.sum(arr1.clone())
            val endTimeNative = System.currentTimeMillis()
            nativeTimeTaken += endTimeNative - startTimeNative

            val startTimeCPP = System.currentTimeMillis()
            val rsltCPP = Operations.sumJNI(arr1.clone())
            val endTimeCPP = System.currentTimeMillis()
            cppTimeTaken += endTimeCPP - startTimeCPP

            val startTimeNeon = System.currentTimeMillis()
            val rsltNeon = Operations.sumNeon(arr1.clone())
            val endTimeNeon = System.currentTimeMillis()
            neonTimeTaken += endTimeNeon - startTimeNeon

            println("-------------------------------------")
            println("Neon time : $neonTimeTaken ms")
            println("C++ time : $cppTimeTaken ms")
            println("Native time : $nativeTimeTaken ms")
            println("rsult Neon : $rsltNeon $rsltCPP $rsltNative")

            val result = TestHelper.compareResults(rsltNative, rsltCPP, rsltNeon)
            assert(result)
        }
    }

    @Test
    fun testSum(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0
        val opTag = "sum"
        val numTrial = 100

        val arrSize = 1024 * 8
        val arr1 = FloatArray(arrSize){ it.toFloat() }

        val startTimeNative = System.currentTimeMillis()
        val rsltNative = Operations.sum(arr1.clone())
        val endTimeNative = System.currentTimeMillis()
        nativeTimeTaken += endTimeNative - startTimeNative

        val startTimeCPP = System.currentTimeMillis()
        val rsltCPP = Operations.sumJNI(arr1.clone())
        val endTimeCPP = System.currentTimeMillis()
        cppTimeTaken += endTimeCPP - startTimeCPP

        val startTimeNeon = System.currentTimeMillis()
        val rsltNeon = Operations.sumNeon(arr1.clone())
        val endTimeNeon = System.currentTimeMillis()
        neonTimeTaken += endTimeNeon - startTimeNeon

        println("-------------------------------------")
        println("Neon time : $neonTimeTaken ms")
        println("C++ time : $cppTimeTaken ms")
        println("Native time : $nativeTimeTaken ms")
        println("rsult Neon : $rsltNeon $rsltCPP $rsltNative")

        val result = rsltNative == rsltCPP && rsltNative == rsltNeon
        assert(result)
    }

    @Test
    fun testAverageOpTimeElapsed(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0
        val numTrial = 10000
        val len = Random.nextInt(5000 - 100, 5000)
        val arr1 = FloatArray(len){ it.toFloat() }

        println("native ${Operations.sum(arr1) / arr1.size}")
        println("cpp ${Operations.averageJNI(arr1)}")
        println("neon ${Operations.averageNeon(arr1)}")

        // neon improves performance by .58
        for(no in 0 until NumTrial){

            val startTimeNative = System.currentTimeMillis()
            val rsltNative = Operations.sum(arr1) / arr1.size
            val endTimeNative = System.currentTimeMillis()
            nativeTimeTaken += endTimeNative - startTimeNative

            val startTimeCPP = System.currentTimeMillis()
            val rsltCPP = Operations.averageJNI(arr1)
            val endTimeCPP = System.currentTimeMillis()
            cppTimeTaken += endTimeCPP - startTimeCPP

            val startTimeNeon = System.currentTimeMillis()
            val rsltNeon = Operations.averageNeon(arr1)
            val endTimeNeon = System.currentTimeMillis()
            neonTimeTaken += endTimeNeon - startTimeNeon
            assert((rsltNative == rsltCPP) && (rsltNative == rsltNeon))
        }

        println("avg time taken neon ($len)   : ${if(neonTimeTaken == 0.0) 0 else neonTimeTaken / numTrial}")
        println("avg time taken cpp ($len)    : ${if(cppTimeTaken == 0.0) 0 else  cppTimeTaken / numTrial}")
        println("avg time taken native ($len) : ${if(nativeTimeTaken == 0.0) 0 else nativeTimeTaken / numTrial}")
    }

    @Test
    fun testAverageOp(){
        var nativeTimeTaken = 0.0
        var cppTimeTaken = 0.0
        var neonTimeTaken = 0.0

        val arr1 = FloatArray(Random.nextInt(2000, 5000)){ it.toFloat() }

        val startTimeNative = System.currentTimeMillis()
        val rsltNative = Operations.sum(arr1.clone()) / arr1.size
        val endTimeNative = System.currentTimeMillis()
        nativeTimeTaken += endTimeNative - startTimeNative

        val startTimeCPP = System.currentTimeMillis()
        val rsltCPP = Operations.averageJNI(arr1.clone())
        val endTimeCPP = System.currentTimeMillis()
        cppTimeTaken += endTimeCPP - startTimeCPP

        val startTimeNeon = System.currentTimeMillis()
        val rsltNeon = Operations.averageNeon(arr1.clone())
        val endTimeNeon = System.currentTimeMillis()
        neonTimeTaken += endTimeNeon - startTimeNeon

        println("$rsltNative $rsltCPP $rsltNeon")
        val result = rsltNative == rsltCPP && rsltNative == rsltNeon
        assert(result)
    }
}