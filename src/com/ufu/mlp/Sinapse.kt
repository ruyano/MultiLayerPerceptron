package com.ufu.mlp

class Sinapse(
    val finalPerceptron: Perceptron,
    val weight: Double = 0.0,
    val gradient: Double? = null
) {

    override fun toString(): String {
        return ("\n\t[Sinapse. Final node number: "
                + finalPerceptron.layerNumber + "/" + finalPerceptron.perceptronNumber
                + ", weight: "
                + weight
                + "]")
    }

    override fun equals(other: Any?): Boolean {
        return finalPerceptron.layerNumber == (other as Sinapse).finalPerceptron.layerNumber &&
                finalPerceptron.perceptronNumber == (other as Sinapse).finalPerceptron.perceptronNumber
    }
}