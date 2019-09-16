import com.ufu.mlp.Layer
import com.ufu.mlp.MultilayerPerceptron
import com.ufu.mlp.Sigmoid
import com.ufu.mlp.Subject
import java.io.File

fun main() {

//    val multilayerPerceptron = MultilayerPerceptron(42)
//    multilayerPerceptron.layers.add(Layer(4, null, multilayerPerceptron))
//    multilayerPerceptron.layers.add(Layer(8, Sigmoid(), multilayerPerceptron))
//    multilayerPerceptron.layers.add(Layer(3, Sigmoid(), multilayerPerceptron))
//
//    for (layer in multilayerPerceptron.layers) {
//        println(layer)
//    }

    val subjects = readFromFile()
    println(subjects.toString())


}

fun readFromFile() : ArrayList<Subject>  {
    val subjects = arrayListOf<Subject>()

    File("training_data.txt").forEachLine {
        val subject = Subject()
        val parts = it.split(",")
        parts.forEach { part ->
            part.toDoubleOrNull()?.let { doubleValue ->
                subject.attributes.add(doubleValue)
            } ?: run {
                when (part) {
                    "Iris-setosa" -> {
                        subject.groupIdentification.add(0.1)
                        subject.groupIdentification.add(0.0)
                        subject.groupIdentification.add(0.0)
                    }
                    "Iris-versicolor" -> {
                        subject.groupIdentification.add(0.0)
                        subject.groupIdentification.add(0.1)
                        subject.groupIdentification.add(0.0)
                    }
                    else -> {
                        subject.groupIdentification.add(0.0)
                        subject.groupIdentification.add(0.0)
                        subject.groupIdentification.add(0.1)
                    }
                }
            }
        }
        subjects.add(subject)
    }
    return subjects
}