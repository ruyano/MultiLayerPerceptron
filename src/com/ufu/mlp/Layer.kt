package com.ufu.mlp

class Layer(
    val activation: Activation = Sigmoid(),
    val perceptrons: ArrayList<Perceptron> = arrayListOf(),
    val number: Int = 0,
    val bias: Perceptron = Perceptron()
) {

    constructor(perceptronsAmount: Int, activation: Activation, multilayerPerceptron: MultilayerPerceptron) : this(
        activation,
        arrayListOf(),
        multilayerPerceptron.layers.size + 1,
        Perceptron()) {
        bias.layerNumber = number
        // inicializa todos os perceptrons desta camada
        for (i in 0 until perceptronsAmount) {
            val perceptron = Perceptron()
            perceptron.layerNumber = number
            perceptron.perceptronNumber = i+1
            perceptrons.add(perceptron)
        }
        // criando as sinapses da camada anterior, conectando esta camada à ela.
        if (multilayerPerceptron.layers.size > 0) {
            // a camada de entrada não tem layer anterior
            val previousLayer = multilayerPerceptron.layers[multilayerPerceptron.layers.size -1]
            for (previousPerceptron in previousLayer.perceptrons) {
                for (currentPerceptron in perceptrons) {
                    previousPerceptron.sinapses.add(Sinapse(currentPerceptron, multilayerPerceptron.getRandom()))
                }
            }
            // Bias da camada anterior
            for (currentPerceptron in perceptrons) {
                previousLayer.bias.sinapses.add(Sinapse(currentPerceptron, multilayerPerceptron.getRandom()))
            }
        } else {
            multilayerPerceptron.firstLayer = this
        }
        multilayerPerceptron.lastLayer = this
    }

    override fun toString(): String {
        return ("\n[Layer. Number : " + number
                + "\nBias: "
                + bias
                + "\nnodes:\n"
                + perceptrons
                + "\n]")
    }
}