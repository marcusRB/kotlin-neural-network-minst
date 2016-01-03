package com.github.winteryoung.mltest

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure
import org.apache.commons.math3.analysis.function.Sigmoid
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector
import java.util.*
import java.util.concurrent.ForkJoinPool

/**
 * @author Winter Young
 * @since 2015/12/19
 */
class NeuralNetwork private constructor(
        private val weights: List<WeightMatrix>,
        private val biases: List<BiasVector>,
        private val learningRate: Double,
        private val gradientChecking: Boolean = false,
        private val debugging: Boolean = false
) {
    constructor(
            layerSizes: List<Int>,
            learningRate: Double
    ) : this(createWeights(layerSizes), createBiases(layerSizes), learningRate)

    init {
        if (weights.size < 2) {
            throw Exception("Layer size at least is 3, but is ${weights.size}")
        }
    }

    val layerSize = weights.size + 1
    private val sigmoidFunction = Sigmoid()
    private val forkJoinPool = ForkJoinPool(Runtime.getRuntime().availableProcessors())

    override fun toString(): String {
        return "biases =\n${biases.joinToString("\n")}\n" +
                "weights =\n${weights.joinToString("\n")}"
    }

    private fun copyWithWeightOfNeuronPerturbed(
            epsilon: Double,
            layer: Int,
            neuron: Int,
            previousLayerNeuron: Int
    ): NeuralNetwork {
        if (layer < 1 || layer > weights.size) {
            throw IndexOutOfBoundsException("layer: $layer, max: ${weights.size}")
        }
        return NeuralNetwork(
                ArrayList(weights).mapIndexed { i, weightMatrix ->
                    if (layer - 1 == i) {
                        weightMatrix.copyWithNeuronPerturbed(epsilon, neuron, previousLayerNeuron)
                    } else {
                        weightMatrix.copy()
                    }
                },
                ArrayList(biases).map { it.copy() },
                learningRate
        )
    }

    private fun copyWithBiasOfNeuronPerturbed(delta: Double, layer: Int, neuron: Int): NeuralNetwork {
        if (layer < 1 || layer > biases.size) {
            throw IndexOutOfBoundsException("layer: $layer, max: ${biases.size}")
        }
        return NeuralNetwork(
                ArrayList(weights).map { it.copy() },
                ArrayList(biases).mapIndexed { i, biasVector ->
                    if (layer - 1 == i) {
                        biasVector.copyWithNeuronPerturbed(delta, neuron)
                    } else {
                        biasVector.copy()
                    }
                },
                learningRate
        )
    }

    fun train(
            trainingData: MutableList<LabeledData>,
            epochs: Int,
            miniBatchSize: Int,
            eachEpoch: (Int) -> Unit = { }
    ) = miniBatchGradientDescent(trainingData, epochs, miniBatchSize, eachEpoch)

    private fun miniBatchGradientDescent(
            trainingData: MutableList<LabeledData>,
            epochs: Int,
            miniBatchSize: Int,
            eachEpoch: (Int) -> Unit
    ) {
        for (epoch in 1..epochs) {
            if (debugging) {
                println("Training epoch $epoch")
            }

            trainingData.shuffle()

            trainingData.split(miniBatchSize).forEachIndexed { batchIndex, batch ->
                TraceEnv.use("batch: $batchIndex") {
                    val forkJoinTask = DataForkJoinTask(batch, weights, biases) { work ->
                        trainMiniBatch(work)
                    }
                    val (weightDecs, biasDecs) = forkJoinPool.invoke(forkJoinTask)
                    val learningRateOfBatch = learningRate / batch.size
                    for ((w, wd) in weights.zip(weightDecs)) {
                        w.matrix -= wd.matrix * learningRateOfBatch
                    }
                    for ((b, bd) in biases.zip(biasDecs)) {
                        b.matrix -= bd.matrix * learningRateOfBatch
                    }
                }
            }

            eachEpoch(epoch)
        }
    }

    private fun trainMiniBatch(batch: List<LabeledData>): Pair<List<WeightMatrix>, List<BiasVector>> {
        val weightDecsOfBatch = weights.map { it.zero() }
        val biasDecsOfBatch = biases.map { it.zero() }
        batch.forEachIndexed { i, labeledData ->
            TraceEnv.use("data index: $i") {
                val (input, actual) = labeledData
                val (weightDecs, biasDecs) = backPropagate(input, actual)
                for ((wdb, wd) in weightDecsOfBatch.zip(weightDecs)) {
                    wdb.matrix += wd.matrix
                }
                for ((bdb, bd) in biasDecsOfBatch.zip(biasDecs)) {
                    bdb.matrix += bd.matrix
                }

                if (debugging) {
                    println("Training done, $TraceEnv")
                }
            }
        }
        return Pair(weightDecsOfBatch, biasDecsOfBatch)
    }

    private fun checkLayer(
            layer: Int,
            input: RealVector,
            actual: RealVector,
            actualPartialWeight: WeightMatrix,
            actualPartialBias: BiasVector
    ) {
        if (gradientChecking.not()) {
            return
        }

        val epsilon = 0.0001
        val delta = 0.001
        val weightMatrix = weights[layer - 1]
        val expectedPartialWeight = weightMatrix.zero()
        for (neuron in 0..weightMatrix.matrix.rowDimension - 1) {
            for (prevLayerNeuron in 0..weightMatrix.matrix.columnDimension - 1) {
                val nn = copyWithWeightOfNeuronPerturbed(epsilon, layer, neuron, prevLayerNeuron)
                val p = nn.predict(input)
                val c = cost(p, actual)

                val nn2 = copyWithWeightOfNeuronPerturbed(-epsilon, layer, neuron, prevLayerNeuron)
                val p2 = nn2.predict(input)
                val c2 = cost(p2, actual)

                val partialCostWeight = (c - c2) / (2 * epsilon)
                expectedPartialWeight.matrix.run {
                    setEntry(neuron, prevLayerNeuron, partialCostWeight)
                }
            }
        }

        checkCloseEnough(expectedPartialWeight.matrix, actualPartialWeight.matrix, delta)

        val biasVector = biases[layer - 1]
        val expectedPartialBias = biasVector.zero()
        for (neuron in 0..biasVector.vector.dimension - 1) {
            val nn = copyWithBiasOfNeuronPerturbed(epsilon, layer, neuron)
            val p = nn.predict(input)
            val c = cost(p, actual)

            val nn2 = copyWithBiasOfNeuronPerturbed(-epsilon, layer, neuron)
            val p2 = nn2.predict(input)
            val c2 = cost(p2, actual)

            val partialCostBias = (c - c2) / (2 * epsilon)
            expectedPartialBias.matrix.run {
                setEntry(neuron, 0, partialCostBias)
            }
        }

        checkCloseEnough(expectedPartialBias.matrix, actualPartialBias.matrix, delta)
    }

    private fun checkCloseEnough(expected: RealMatrix, actual: RealMatrix, delta: Double) {
        val rows = expected.rowDimension
        if (rows != actual.rowDimension) {
            throw Exception("Row dimensions diff, expected: $rows," +
                    " actual: ${actual.rowDimension}")
        }
        val cols = expected.columnDimension
        if (cols != actual.columnDimension) {
            throw Exception("Column dimension diff, expected: $cols," +
                    " actual: ${actual.columnDimension}")
        }

        for (row in 0..rows - 1) {
            TraceEnv.use("row: $row") {
                for (col in 0..cols - 1) {
                    TraceEnv.use("col: $col") {
                        val exp = expected.getEntry(row, col)
                        val act = actual.getEntry(row, col)
                        if (act.equals(exp, delta).not()) {
                            throw Exception("Gradient check failed, expected: $exp, actual: $act")
                        }
                    }
                }
            }
        }
    }

    fun predict(input: RealVector) = feedForward(input)[0].getx(-1)

    private fun feedForward(input: RealVector): List<List<RealVector>> {
        val activations = ArrayList<RealVector>().apply { add(input) }
        val weightedInputs = ArrayList<RealVector>()
        var activation = input
        for ((weight, bias) in weights.zip(biases)) {
            val weightedInput = weight.matrix * activation + bias.vector
            weightedInputs.add(weightedInput)
            activation = activate(weightedInput)
            activations.add(activation)
        }
        return listOf(activations, weightedInputs)
    }

    private fun backPropagate(input: RealVector, actual: RealVector): Gradient {
        val (activations, weightedInputs) = feedForward(input)
        var error: RealVector? = null

        val weightDecs = weights.map { it.zero() }
        fun updateWeightDecs(layer: Int) {
            weightDecs.getx(layer).matrix = error!!.outerProduct(activations.getx(layer - 1))
        }

        val biasDecs = biases.map { it.zero() }
        fun updateBiasDecs(layer: Int) {
            biasDecs.getx(layer).matrix = error!!.toRealMatrix()
        }

        TraceEnv.use("layer: -1") {
            error = run {
                val activationDerivative = activateDerivative(weightedInputs.getx(-1))
                val costDerivative = costDerivative(activations.getx(-1), actual)
                costDerivative * activationDerivative
            }

            updateWeightDecs(-1)
            updateBiasDecs(-1)
        }

        checkLayer(xindex(-1, layerSize), input, actual, weightDecs.getx(-1), biasDecs.getx(-1))

        for (layer in -2 downTo -weightDecs.size) {
            TraceEnv.use("layer: $layer") {
                val ad = activateDerivative(weightedInputs.getx(layer))
                error = weights.getx(layer + 1).matrix.transpose() * error!! * ad
                updateWeightDecs(layer)
                updateBiasDecs(layer)
                checkLayer(xindex(layer, layerSize), input, actual, weightDecs.getx(layer), biasDecs.getx(layer))
            }
        }

        return Gradient(weightDecs, biasDecs)
    }

    private fun cost(predication: RealVector, actual: RealVector): Double {
        val vecCost = 0.5 * (actual - predication).pow(2.0)
        return vecCost.toArray().sum()
    }

    private fun costDerivative(activation: RealVector, actual: RealVector): RealVector {
        return activation - actual
    }

    private fun activate(weightedInput: RealVector) = sigmoid(weightedInput)

    private fun activateDerivative(weightedInput: RealVector): RealVector {
        weightedInput.map {
            val y = sigmoidFunction.value(DerivativeStructure(1, 1, it))
            y.getPartialDerivative(1)
        }
        val a = sigmoid(weightedInput)
        return a * (1.0 - a)
    }

    private fun sigmoid(z: RealVector): RealVector {
        return z.map { sigmoidFunction.value(it) }
    }

    private data class Gradient(
            val weightDecs: List<WeightMatrix>,
            val biasDecs: List<BiasVector>
    )

    companion object {
        private fun createWeights(layerSizes: List<Int>): List<WeightMatrix> {
            val leftLayerSizes = layerSizes.subList(0, layerSizes.size - 1)
            val rightLayerSizes = layerSizes.subList(1, layerSizes.size)
            return leftLayerSizes.zip(rightLayerSizes).map {
                val (width, height) = it
                val nd = NormalDistribution()
                WeightMatrix(height, width) {
                    nd.sample()
                }
            }
        }

        private fun createBiases(layerSizes: List<Int>): List<BiasVector> {
            return layerSizes.subList(1, layerSizes.size).map { width ->
                val nd = NormalDistribution()
                BiasVector(width) {
                    nd.sample()
                }
            }
        }
    }
}
