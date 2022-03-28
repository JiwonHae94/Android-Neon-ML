package com.jiwon.neon_simd.helper

import android.util.Log
import androidx.core.util.toRange
import kotlin.random.Random

object TestHelper {
    const val MarginOfError = 0.02f

    fun generateRandomDoubleArray(len : Int) : Pair<DoubleArray, DoubleArray>{
        val arr1 = Array(len){ Random.nextDouble() }
        val arr2 = Array(len){ Random.nextDouble() }
        return Pair(arr1.toDoubleArray(), arr2.toDoubleArray())
    }

    fun generateRandomFloatArray(len : Int) : Pair<FloatArray, FloatArray>{
        val arr1 = Array(len){ Random.nextFloat() }
        val arr2 = Array(len){ Random.nextFloat() }
        return Pair(arr1.toFloatArray(), arr2.toFloatArray())
    }

    /**
     * @param marginOfDifference : difference acceptance
     */
    fun <T : Number>compareResults(vararg t : T, marginOfDifference : Float = MarginOfError) : Boolean {
        var `val` = t.map{ it.toFloat() }
        val ranges = `val`.map {
            val diff = it * marginOfDifference
            (it - diff)..(it + diff)
        }

        println(ranges.joinToString(" "))

        for(i in `val`){
            for(r in ranges){
                if (i in r){
                    continue
                }else{
                    return false
                }
            }
        }

        return true
    }

}