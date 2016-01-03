package com.github.winteryoung.mltest

import java.util.*

fun main(args: Array<String>) {
    val mnistFolder = "mnist"
    val trainingData = readMnistTrainingData(mnistFolder)
    val testData = readMnistTestData(mnistFolder)

    val nn = NeuralNetwork(layerSizes = listOf(784, 30, 10), learningRate = 3.0)
    nn.train(trainingData.map { it.toLabeledData() }.toArrayList(), epochs = 300, miniBatchSize = 100) {
        val failures = ArrayList<Pair<LabeledDigitImage, Byte>>()
        for (labeledDigitImage in testData) {
            val prediction = nn.predict(labeledDigitImage.toLabeledData().data).maxIndex.toByte()
            if (prediction != labeledDigitImage.label) {
                failures.add(labeledDigitImage to prediction)
            }
        }
        val successRate = 1 - (failures.size.toDouble() / testData.size)
        val successPercent = "%.2f".format(successRate * 100)
        println("Epoch $it done, success rate: $successPercent%")
    }
}