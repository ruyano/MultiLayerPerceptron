package com.ufu.mlp

import kotlin.math.pow
import kotlin.random.Random

class MultilayerPerceptron(
    val layers: ArrayList<Layer>,
    private val random: Random,
    var firstLayer: Layer,
    var lastLayer: Layer
) {
    constructor(seed: Int) : this(
        arrayListOf(),
        Random(seed),
        Layer(),
        Layer()
    )

    // função que executa a passagem dos dados pelo mlp
    fun forwardPass(input: ArrayList<Double>) : ArrayList<Double> {
        for (layer in layers) {
            // camada de entrada
            if (layer == firstLayer) {
                initializeInputLayer(layer, input)
            } else {
                // demais camadas
                executeMlp(layer)
            }
        }
        return getOutputValues()
    }

    // executa a passagem dos dados de entrada pelo MLP
    private fun executeMlp(layer: Layer) {
        for (perceptron in layer.perceptrons) {
            var finalValue = 0.0
            val previousLayer = layers[(layers.indexOf(layer) - 1)]
            for (previousLayerPerceptron in previousLayer.perceptrons) {
                val sinapse = getSinapse(previousLayerPerceptron, perceptron)
                finalValue += previousLayerPerceptron.value * sinapse.weight
            }
            // Bias anterior
            val sinapseBias = getSinapse(previousLayer.bias, perceptron)
            finalValue += sinapseBias.weight
            perceptron.input = finalValue
            perceptron.value = layer.activation.execute(perceptron.input)
        }
    }

    // recupera os valores da camada de saida como output de resposta
    private fun getOutputValues(): ArrayList<Double> {
        val outputValue = arrayListOf<Double>()
        for (perceptron in lastLayer.perceptrons) {
            outputValue.add(perceptron.value)
        }
        return outputValue
    }

    // recupera a sinapse entre 2 perceptrons
    private fun getSinapse(origem: Perceptron, destino: Perceptron): Sinapse {
        val indexSin = origem.sinapses.indexOf(Sinapse(destino))
        return origem.sinapses[indexSin]
    }

    // inicializa camada de entrada com os valores de entrada
    private fun initializeInputLayer(layer: Layer, input: ArrayList<Double>) {
        for ((i, perceptron) in layer.perceptrons.withIndex()) {
            perceptron.input = input[i]
            perceptron.value = input[i]
        }
    }

    // função que realiza o backPropagation para cálculo dos novos pesos
    fun backPropagation(target: ArrayList<Double>, learningRate: Double) {
        // TODO
    }

    // função que executa o treinamento da rede mlp
    fun fit(dataset: ArrayList<Subject>,
            epochs: Int,
            learningRate: Double) {

        for (epoch in 0 until epochs) {
            var mse = 0.0
            for (subject in dataset) {
                val outputs = forwardPass(subject.attributes)
                for (z in 0 until outputs.size) {
                    mse += (subject.groupIdentification[z] - outputs[z]).pow(2)
                }
                backPropagation(subject.groupIdentification, learningRate)
            }
            mse /= dataset.size
            println("Epoch: $epoch MSE: $mse")
        }

    }

    // gera um número randomico entre -1 e 1
    fun getRandom(): Double {
        return -1.0 + (1.0 - (-1.0)) * random.nextDouble();
    }

}