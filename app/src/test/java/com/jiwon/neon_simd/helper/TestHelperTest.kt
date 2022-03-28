package com.jiwon.neon_simd.helper

import org.junit.Assert.*
import org.junit.Test

class TestHelperTest{
    @Test
    fun `test result evaluator1`(){
        val testResult = TestHelper.compareResults(1000f, 1300f, marginOfDifference = 0.3f)
        assert(testResult)
    }

    @Test
    fun `test result evaluator2`(){
        val testResult = TestHelper.compareResults(1000f, 1100f, marginOfDifference = 0.5f)
        assert(testResult)
    }

    @Test
    fun `test result evaluator3`(){
        val testResult = TestHelper.compareResults(1000f, 900f, marginOfDifference = 0.4f)
        assert(testResult)
    }

}