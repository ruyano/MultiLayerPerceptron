package com.ufu.mlp

import kotlin.math.pow

class Sigmoid : Activation() {

    override fun execute(input: Double): Double {
        return (1 / (1 + Math.E.pow((-1 * input))))
    }

    override fun derivate(parameter: Double): Double {
        return parameter * (1 - parameter)
    }

}