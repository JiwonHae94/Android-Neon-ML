package com.jiwon.neon_simd

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<TextView>(R.id.)
    }

    external fun stringFromJNI() : String

    init{
        System.loadLibrary("native-lib")
    }
}