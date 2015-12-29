package com.github.winteryoung.mltest

fun main(args: Array<String>) {
    val nn = NeuralNetwork(listOf(2, 3, 1))
    println(nn)

    val mnistFolder = """D:\code\machine-learning-test\mnist"""
    val trainingData = readMnistTrainingData(mnistFolder)
    val testData = readMnistTestData(mnistFolder)

    showImage(testData[0].data.toBufferedImage())
}

