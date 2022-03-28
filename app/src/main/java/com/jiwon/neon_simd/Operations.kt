package com.jiwon.neon_simd

import android.util.Log

object Operations {
    // formulae : x dot y / ||x|| * ||y||
    fun cosineSimilarity(arr1 : FloatArray, arr2 : FloatArray): Double{
        var sumProduct = 0f
        var sumOfsqA = 0.0
        var sumOfsqB = 0.0

        for(i in 0 until arr1.size){
            sumProduct += arr1.get(i) * arr2.get(i)
            sumOfsqA += Math.pow(arr1.get(i).toDouble(), 2.0)
            sumOfsqB += Math.pow(arr2.get(i).toDouble(), 2.0)
        }
        return sumProduct / (Math.sqrt(sumOfsqA) * Math.sqrt(sumOfsqB))
    }

    fun dotMatrix(arr1 : FloatArray, arr2 : FloatArray) = arr1.zip(arr2).sumOf { pair ->
        (pair.first * pair.second).toDouble()
    }


    external fun cosineSimilarityCPP(arr1 : FloatArray, arr2 : FloatArray) : Float

    external fun cosineSimilarityNeon(arr1 : FloatArray, arr2 : FloatArray) : Float

    external fun dotNeon(arr1 : FloatArray, arr2 : FloatArray) : Float

    external fun dotJNI(arr1 : FloatArray, arr2 : FloatArray) : Float

    init{
        System.loadLibrary("jni_array_op")
    }
}