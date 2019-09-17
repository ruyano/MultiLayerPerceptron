package com.ufu.mlp

import kotlin.math.pow
import kotlin.math.roundToInt
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

    constructor() : this(
            arrayListOf(),
            Random,
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
            layer.activation?.let {
                perceptron.value = it.execute(perceptron.input)
            }
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

        // registra a saida sa ultima camada e o erro comparado ao target
        val outputErrors = arrayListOf<Double>()
        val outputs = arrayListOf<Double>()
        for ((i, perceptron) in lastLayer.perceptrons.withIndex()) {
            outputErrors.add(i, perceptron.value - target[i])
            outputs.add(i, perceptron.value)
        }

        // percorrendo as camadas menos a ultima em ordem reversa
        val reverseOrderLayers = layers.asReversed()
        for (currentLayer in reverseOrderLayers) {
            // não realiza na ultima pois já foi validado anteriormente
            if (currentLayer != lastLayer) {

                // inicializa componentes basicos
                val lastLayerIndex = layers.indexOf(lastLayer)
                val currentLayerIndex = layers.indexOf(currentLayer)
                val nextLayer = layers[layers.indexOf(currentLayer) + 1]

                // percorre os percetros da vamada atual
                for (currentPerceptron in currentLayer.perceptrons) {
                    if (currentLayerIndex == (lastLayerIndex - 1)) {
                        // se for a penultima camada
                        for (sinapse in currentPerceptron.sinapses) {
                            // calcula o gradiente da penultima camada
                            val erro = outputErrors[sinapse.finalPerceptron.perceptronNumber - 1]
                            nextLayer.activation?.let {
                                sinapse.gradient = erro * it.derivate(sinapse.finalPerceptron.value) * currentPerceptron.value
                            }
                        }
                    } else {
                        // demais camadas (antepenultima para traz)
                        for (sinapse in currentPerceptron.sinapses) {
                            var finalValue = 0.0
                            for (sinapse2 in sinapse.finalPerceptron.sinapses) {
                                val deltaz = outputErrors[sinapse2.finalPerceptron.perceptronNumber - 1] * outputs[sinapse2.finalPerceptron.perceptronNumber - 1] * (1 - outputs[sinapse2.finalPerceptron.perceptronNumber - 1])
                                finalValue += deltaz * sinapse2.weight
                            }
                            nextLayer.activation?.let {
                                sinapse.gradient = finalValue * it.derivate(sinapse.finalPerceptron.value) * currentPerceptron.value
                            }
                        }
                    }
                }

                // peso do bias
                if (currentLayerIndex == (lastLayerIndex - 1)) {
                    // se for a penultima camada
                    for (sinapse in currentLayer.bias.sinapses) {
                        val erro = sinapse.finalPerceptron.value - target[sinapse.finalPerceptron.perceptronNumber - 1]
                        currentLayer.activation?.let {
                            sinapse.gradient = erro * it.derivate(sinapse.finalPerceptron.value)
                        }
                    }
                } else {
                    // demais camadas (antepenultima para traz)
                    for (sinapse in currentLayer.bias.sinapses) {
                        var finalValue = 0.0
                        for (sinapse2 in sinapse.finalPerceptron.sinapses) {
                            val deltaz = outputErrors[sinapse2.finalPerceptron.perceptronNumber - 1] * outputs[sinapse2.finalPerceptron.perceptronNumber - 1] * (1 - outputs[sinapse2.finalPerceptron.perceptronNumber - 1])
                            finalValue += deltaz * sinapse2.weight
                        }
                        nextLayer.activation?.let {
                            sinapse.gradient = finalValue * it.derivate(sinapse.finalPerceptron.value)
                        }
                    }
                }
            }
            // atualiza os pesos
            for (layer in layers) {
                for (perceptron in layer.perceptrons) {
                    for (sinapse in perceptron.sinapses) {
                        sinapse.weight = sinapse.weight - learningRate * sinapse.gradient
                    }
                }
            }
        }
    }

    // função que executa o treinamento da rede mlp
    fun fit(dataset: ArrayList<Subject>, epochs: Int, learningRate: Double) {
        for (epoch in 0 until epochs) {
            var mse = 0.0
            for (subject in dataset) {
                val outputs = forwardPass(subject.attributes)
                println("Entrada: ${subject.attributes} | Esperado: ${senioridade(subject.targetResult)}")
                println("Calculado: [${outputs[0].roundToInt()}, ${outputs[1].roundToInt()}, ${outputs[2].roundToInt()}] = ${senioridade(outputs)}")
                for ((i,z) in outputs.withIndex()) {
                    mse += (subject.targetResult[i] - z).pow(2)
                }
                backPropagation(subject.targetResult, learningRate)
            }
            mse /= dataset.size
            println("Epoch: $epoch MSE: $mse")
        }
    }

    fun executeAvaliation(subjectsForAvaliation: ArrayList<Subject>) {
        var erros = 0
        var contagem = 0
        for (subject in subjectsForAvaliation) {
            val result = forwardPass(subject.attributes)
            println("Entrada: ${subject.attributes} | Esperado: ${senioridade(subject.targetResult)}")
            println("Calculado: [${result[0].roundToInt()}, ${result[1].roundToInt()}, ${result[2].roundToInt()}] = ${senioridade(result)}")
            contagem++
            var erro = false
            for ((i, target) in subject.targetResult.withIndex()) {
                erro = result[i].roundToInt() != target.roundToInt()
            }
            if (erro) {
                erros++
            }
        }
        val acuracia = 100.0 - erros.toDouble() / contagem.toDouble() * 100
        println("Testes: $contagem erros: $erros acurácia: $acuracia")
    }

    private fun senioridade(array: ArrayList<Double>) : String {
        val c1 = array[0].roundToInt()
        val c2 = array[1].roundToInt()
        val c3 = array[2].roundToInt()
        return if (c1 == 1 && c2 == 0 && c3 == 0) {
            "Senior"
        } else if (c1 == 0 && c2 == 1 && c3 == 0) {
            "Pleno"
        } else {
            "Junior"
        }
    }

    // gera um número randomico entre -1 e 1
    fun getRandom(): Double {
        return -1.0 + (1.0 - (-1.0)) * random.nextDouble();
    }

}