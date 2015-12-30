package com.github.winteryoung.mltest

import java.util.*

fun main(args: Array<String>) {
    val mnistFolder = "mnist"
    val trainingData = readMnistTrainingData(mnistFolder)
    val testData = readMnistTestData(mnistFolder)

    val nn = NeuralNetwork(listOf(784, 30, 10), 3.0)
    nn.train(trainingData.map { it.toLabeledData() }.toArrayList(), 30, 10)
    println("nn: $nn")

    val failures = ArrayList<Pair<LabeledDigitImage, Byte>>()
    for (labeledDigitImage in testData) {
        val prediction = nn.predict(labeledDigitImage.toLabeledData().data).maxIndex.toByte()
        if (prediction != labeledDigitImage.label) {
            failures.add(labeledDigitImage to prediction)
        }
    }
    println("Success rate: ${1 - failures.size.toDouble() / testData.size}")

    for ((labeledDigitImage, predication) in failures) {
        println("Actual: ${labeledDigitImage.label}")
        showImage(labeledDigitImage.data.toBufferedImage())
    }
}