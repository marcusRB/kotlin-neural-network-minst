package com.github.winteryoung.mltest

import java.util.*

fun main(args: Array<String>) {
    val mnistFolder = "mnist"
    val trainingData = readMnistTrainingData(mnistFolder)
    val testData = readMnistTestData(mnistFolder)

    val nn = NeuralNetwork(layerSizes = listOf(784, 30, 10), learningRate = 3.0)
    nn.train(trainingData.map { it.toLabeledData() }.toArrayList(), epochs = 60, miniBatchSize = 10)
    println("nn: $nn")

    val failures = ArrayList<Pair<LabeledDigitImage, Byte>>()
    for (labeledDigitImage in testData) {
        val prediction = nn.predict(labeledDigitImage.toLabeledData().data).maxIndex.toByte()
        if (prediction != labeledDigitImage.label) {
            failures.add(labeledDigitImage to prediction)
        }
    }
    println("Success rate: ${1 - failures.size.toDouble() / testData.size}")

    for ((labeledDigitImage, predication) in failures.take(3)) {
        println("Actual: ${labeledDigitImage.label}, predication: $predication")
        showImage(labeledDigitImage.data.toBufferedImage())
    }
}