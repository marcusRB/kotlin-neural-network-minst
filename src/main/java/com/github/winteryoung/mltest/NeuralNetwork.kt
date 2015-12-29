package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.RealVector
import java.util.*

/**
 * @author Winter Young
 * @since 2015/12/19
 */
class NeuralNetwork(layerSizes: List<Int>) {
    private val biases: List<BiasVector>
    private val weights: List<WeightMatrix>

    init {
        biases = layerSizes.subList(1, layerSizes.size).map { width ->
            BiasVector(width)
        }
        weights = run {
            val leftLayerSizes = layerSizes.subList(0, layerSizes.size - 1)
            val rightLayerSizes = layerSizes.subList(1, layerSizes.size)
            leftLayerSizes.zip(rightLayerSizes).map {
                val (width, height) = it
                WeightMatrix(height, width)
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
            trainingData: MutableList<Pair<Double, Double>>,
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

    private fun updateMiniBatch(batch: List<Pair<Double, Double>>, learningRate: Double) {
    }
}