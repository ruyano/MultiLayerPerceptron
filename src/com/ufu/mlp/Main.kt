import com.ufu.mlp.Layer
import com.ufu.mlp.MultilayerPerceptron
import com.ufu.mlp.Sigmoid
import com.ufu.mlp.Subject
import java.io.File

fun main() {

    // criar rede neural mlp com 3 camadas, camada de entrada com 4 perceptrons, camada interna com 8 perceptrons e camada de saida com 3 perceptrons
    val multilayerPerceptron = MultilayerPerceptron(42)
    multilayerPerceptron.layers.add(Layer(4, null, multilayerPerceptron))
    multilayerPerceptron.layers.add(Layer(10, Sigmoid(), multilayerPerceptron))
    multilayerPerceptron.layers.add(Layer(3, Sigmoid(), multilayerPerceptron))

    // imprimir dados da rede gerada
    for (layer in multilayerPerceptron.layers) {
        println(layer)
    }

    // le sujeitos para o treinamento do arquivo
    val subjectsForTraining = readFromFile("training_data.txt")

    // executa treinamento
    val epochs = 1000
    val learningRate = 0.01
    multilayerPerceptron.fit(subjectsForTraining, epochs, learningRate)

    // le sujeitos para o treinamento do arquivo
    val subjectsForAvaliation = readFromFile("avaliation_data.txt")

    //executa avaliação
    multilayerPerceptron.executeAvaliation(subjectsForAvaliation)
}

fun readFromFile(fileName: String) : ArrayList<Subject>  {
    val subjects = arrayListOf<Subject>()

    File(fileName).forEachLine {
        val subject = Subject()
        val parts = it.split(",")
        parts.forEach { part ->
            part.toDoubleOrNull()?.let { doubleValue ->
                subject.attributes.add(doubleValue)
            } ?: run {
                when (part) {
                    "Senior" -> {
                        subject.targetResult.add(1.0)
                        subject.targetResult.add(0.0)
                        subject.targetResult.add(0.0)
                    }
                    "Pleno" -> {
                        subject.targetResult.add(0.0)
                        subject.targetResult.add(1.0)
                        subject.targetResult.add(0.0)
                    }
                    else -> {
                        subject.targetResult.add(0.0)
                        subject.targetResult.add(0.0)
                        subject.targetResult.add(1.0)
                    }
                }
            }
        }
        subjects.add(subject)
    }
    return subjects
}