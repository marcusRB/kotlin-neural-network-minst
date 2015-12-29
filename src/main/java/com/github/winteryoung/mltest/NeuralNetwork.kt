package com.github.winteryoung.mltest

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.linear.RealVector
import java.util.*

/**
 * @author Winter Young
 * @since 2015/12/19
 */
class NeuralNetwork(layerSizes: List<Int>) {
    private var weights: List<WeightMatrix>
    private var biases: List<BiasVector>

    init {
        biases = layerSizes.subList(1, layerSizes.size).map { width ->
            val nd = NormalDistribution()
            BiasVector(width) {
                nd.sample()
            }
        }
        weights = run {
            val leftLayerSizes = layerSizes.subList(0, layerSizes.size - 1)
            val rightLayerSizes = layerSizes.subList(1, layerSizes.size)
            leftLayerSizes.zip(rightLayerSizes).map {
                val (width, height) = it
                val nd = NormalDistribution()
                WeightMatrix(height, width) {
                    nd.sample()
                }
            }
        }
    }

    override fun toString(): String {
        return "biases =\n${biases.joinToString("\n")}\n" +
                "weights =\n${weights.joinToString("\n")}"
    }

    private fun feedForward(inputs: RealVector): RealVector {
        var vec = inputs
        for ((bias, weight) in biases.zip(weights)) {
            val weightedInput = weight.multiply(vec).add(bias.vector)
            vec = sigmoid(weightedInput)
        }
        return vec
    }

    private fun sigmoid(z: RealVector): RealVector {
        fun s(z: Double) = 1.0 / (1.0 + Math.exp(-z))
        return z.map {
            s(it)
        }
    }

    private fun miniBatchGradientDescent(
            trainingData: MutableList<LabeledData>,
            epochs: Int,
            miniBatchSize: Int,
            learningRate: Double,
            testData: List<Pair<Double, Double>>? = null
    ) {
        for (epoch in 1..epochs) {
            Collections.shuffle(trainingData)
            val batches = trainingData.split(miniBatchSize)
            for (batch in batches) {
                updateMiniBatch(batch, learningRate)
            }

            if (testData == null) {
                println("Epoch $epoch complete.")
            } else {
                val evaluation = evaluate(testData)
                println("Epoch $epoch: $evaluation / ${testData.size}")
            }
        }
    }

    private fun evaluate(testData: List<Pair<Double, Double>>): Double {
        return 0.0
    }

    private fun updateMiniBatch(batch: List<LabeledData>, learningRate: Double) {
        val weightDecsOfBatch = weights.map { it.zero() }
        val biasDecsOfBatch = biases.map { it.zero() }
        for (labeledData in batch) {
            val (weightDecs, biasDecs) = backPropagate(labeledData)
            for ((wdb, wd) in weightDecsOfBatch.zip(weightDecs)) {
                wdb.matrix = wdb.matrix.add(wd.matrix)
            }
            for ((bdb, bd) in biasDecsOfBatch.zip(biasDecs)) {
                bdb.matrix = bdb.matrix.add(bd.matrix)
            }
        }

        val learningRateOfBatch = learningRate / batch.size
        weights = ArrayList<WeightMatrix>().apply {
            for ((w, wd) in weights.zip(weightDecsOfBatch)) {
                w.matrix = w.matrix.subtract(wd.matrix.scalarMultiply(learningRateOfBatch))
            }
        }
        biases = ArrayList<BiasVector>().apply {
            for ((b, bd) in biases.zip(biasDecsOfBatch)) {
                b.matrix = b.matrix.subtract(bd.matrix.scalarMultiply(learningRateOfBatch))
            }
        }
    }

    private fun backPropagate(labeledData: LabeledData): Gradient {
        throw Exception()
    }

    private data class Gradient(
            val weightDecs: List<WeightMatrix>,
            val biasDecs: List<BiasVector>
    )
}