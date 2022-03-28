package com.jiwon.neon_simd

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.jiwon.neon_simd.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    lateinit var binding : ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        findViewById<TextView>(R.id.log_msg).text = dotProduct()
    }


    external fun dotProduct() : String


    init{
        System.loadLibrary("native-lib")
    }
}