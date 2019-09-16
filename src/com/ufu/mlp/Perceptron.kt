package com.ufu.mlp

class Perceptron(
    var layerNumber: Int? = 0,
    var perceptronNumber: Int = 0,
    val sinapses: ArrayList<Sinapse> = arrayListOf(),
    var input: Double = 0.0,
    var value: Double = 0.0) {

    override fun toString(): String {
        return ("\n\t[Node. Layer: "
                + layerNumber
                + ", perceptronNumber: "
                + perceptronNumber
                + ", input: "
                + input
                + ", value: "
                + value
                + " sinapses: "
                + sinapses
                + "]")
    }
}