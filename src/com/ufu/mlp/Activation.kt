package com.ufu.mlp

abstract class Activation {
    abstract fun execute(input: Double) : Double
    abstract fun derivate(parameter: Double) : Double
}