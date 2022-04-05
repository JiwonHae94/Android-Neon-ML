package com.jiwon.neon_simd

import android.util.Log
import com.jiwon.neon_simd.helper.TestHelper
import java.util.*
import kotlin.math.exp
import kotlin.random.Random

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

    // softmax 1d
    fun softmax(arr1 : FloatArray) : FloatArray{
        val _max = arr1.maxOrNull() ?: return floatArrayOf()
        val exp_x = arr1.map { exp(it - _max) }
        val _sum = exp_x.sum()
        return exp_x.map { it / _sum }.toFloatArray()
    }

    fun dotMatrix(arr1 : FloatArray, arr2 : FloatArray) = arr1.zip(arr2).sumOf { pair ->
        (pair.first * pair.second).toDouble()
    }

    fun sum(arr1 : FloatArray) = arr1.sum()

    external fun cosineSimilarityCPP(arr1 : FloatArray, arr2 : FloatArray) : Float

    external fun cosineSimilarityNeon(arr1 : FloatArray, arr2 : FloatArray) : Float

    external fun dotNeon(arr1 : FloatArray, arr2 : FloatArray) : Float

    external fun dotJNI(arr1 : FloatArray, arr2 : FloatArray) : Float

    external fun softmaxJNI(arr1 : FloatArray) : FloatArray

    external fun softmaxNeon(arr1 : FloatArray) : FloatArray

    external fun sumJNI(arr1 : FloatArray): Float

    external fun sumNeon(arr1 : FloatArray) : Float

    external fun averageJNI(arr1 : FloatArray): Float

    external fun averageNeon(arr1 : FloatArray) : Float

    init{
        System.loadLibrary("neon_op")
    }
}