package com.github.winteryoung.mltest

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.linear.RealVector
import java.util.*

/**
 * @author Winter Young
 * @since 2015/12/19
 */
class NeuralNetwork(
        private val weights: List<WeightMatrix>,
        private val biases: List<BiasVector>,
        private val learningRate: Double
) {
    constructor(
            layerSizes: List<Int>,
            learningRate: Double
    ) : this(createWeights(layerSizes), createBiases(layerSizes), learningRate)

    override fun toString(): String {
        return "biases =\n${biases.joinToString("\n")}\n" +
                "weights =\n${weights.joinToString("\n")}"
    }

    fun copy(deltaInWeights: Double, deltaInBiases: Double): NeuralNetwork {
        return NeuralNetwork(
                ArrayList(weights).map { it.copy(deltaInWeights) },
                ArrayList(biases).map { it.copy(deltaInBiases) },
                learningRate
        )
    }

    fun train(
            trainingData: MutableList<LabeledData>,
            epochs: Int,
            miniBatchSize: Int
    ) = miniBatchGradientDescent(trainingData, epochs, miniBatchSize)

    private fun miniBatchGradientDescent(
            trainingData: MutableList<LabeledData>,
            epochs: Int,
            miniBatchSize: Int
    ) {
        for (epoch in 1..epochs) {
            println("Training $epoch")
            Collections.shuffle(trainingData)
            val batches = trainingData.split(miniBatchSize)
            for (batch in batches) {
                updateMiniBatch(batch, learningRate)
            }
        }
    }

    private fun updateMiniBatch(batch: List<LabeledData>, learningRate: Double) {
        val weightDecsOfBatch = weights.map { it.zero() }
        val biasDecsOfBatch = biases.map { it.zero() }
        for (labeledData in batch) {
            val (weightDecs, biasDecs) = backPropagate(labeledData)
            for ((wdb, wd) in weightDecsOfBatch.zip(weightDecs)) {
                wdb.matrix += wd.matrix
            }
            for ((bdb, bd) in biasDecsOfBatch.zip(biasDecs)) {
                bdb.matrix += bd.matrix
            }
        }

        val learningRateOfBatch = learningRate / batch.size
        for ((w, wd) in weights.zip(weightDecsOfBatch)) {
            w.matrix -= wd.matrix * learningRateOfBatch
        }
        for ((b, bd) in biases.zip(biasDecsOfBatch)) {
            b.matrix -= bd.matrix * learningRateOfBatch
        }
    }

    fun predict(input: RealVector) = feedForward(input)[0].getx(-1)

    private fun feedForward(input: RealVector): List<List<RealVector>> {
        val activations = ArrayList<RealVector>().apply { add(input) }
        val weightedInputs = ArrayList<RealVector>().apply { add(input) }
        var activation = input
        for ((weight, bias) in weights.zip(biases)) {
            val weightedInput = weight.matrix * activation + bias.vector
            weightedInputs.add(weightedInput)
            activation = activate(weightedInput)
            activations.add(activation)
        }
        return listOf(activations, weightedInputs)
    }

    private fun backPropagate(labeledData: LabeledData): Gradient {
        val (input, actual) = labeledData
        val (activations, weightedInputs) = feedForward(input)
        var error = run {
            val activationDerivative = activateDerivative(weightedInputs.getx(-1))
            val costDerivative = costDerivative(activations.getx(-1), actual)
            costDerivative * activationDerivative
        }

        val weightDecs = weights.map { it.zero() }
        weightDecs.getx(-1).matrix = error.outerProduct(activations.getx(-2))

        val biasDecs = biases.map { it.zero() }
        biasDecs.getx(-1).matrix = error.toRealMatrix()

        for (layer in -2 downTo -weightDecs.size) {
            val ad = activateDerivative(weightedInputs.getx(layer))
            error = weights.getx(layer + 1).matrix.transpose() * error * ad
            weightDecs.getx(layer).matrix = error.outerProduct(activations.getx(layer - 1))
            biasDecs.getx(layer).matrix = error.toRealMatrix()
        }

        return Gradient(weightDecs, biasDecs)
    }

    private fun costDerivative(activation: RealVector, actual: RealVector): RealVector {
        return activation - actual
    }

    private fun activate(weightedInput: RealVector) = sigmoid(weightedInput)

    private fun activateDerivative(weightedInput: RealVector): RealVector {
        val a = sigmoid(weightedInput)
        return a * (1.0 - a)
    }

    private fun sigmoid(z: RealVector): RealVector {
        return 1.0 / exp(-z) + 1.0
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