package com.github.winteryoung.mltest

import java.awt.Canvas
import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.awt.image.BufferedImage
import java.io.DataInputStream
import java.io.FileInputStream
import java.nio.file.Paths
import java.util.*
import javax.swing.JFrame
import javax.swing.SwingUtilities

fun readMnistImageFile(path: String): List<Image> {
    FileInputStream(path).buffered().use { inputStream ->
        val dis = DataInputStream(inputStream)
        dis.readInt().run {
            if (this != 0x00000803) {
                throw Exception("Magic number expected for file $path")
            }
        }
        val imageNumber = dis.readInt()

        val imageHeight = dis.readInt()
        val imageWidth = dis.readInt()

        fun genRgb(grayScale: Int): Int = (255 - grayScale).let {
            Color(it, it, it).rgb
        }

        return ArrayList<Image>().apply {
            for (i in 1..imageNumber) {
                val data = IntArray(imageWidth * imageHeight)
                for (h in 0..imageHeight - 1) {
                    for (w in 0..imageWidth - 1) {
                        val rgb = genRgb(dis.readUnsignedByte())
                        data[h * imageWidth + w] = rgb
                    }
                }
                add(Image(imageWidth, imageHeight, data))
            }
        }
    }
}

fun readMnistLabelFile(path: String): List<Byte> {
    FileInputStream(path).buffered().use { inputStream ->
        val dis = DataInputStream(inputStream)
        dis.readInt().run {
            if (this != 0x00000801) {
                throw Exception("Magic number expected for file $path")
            }
        }
        val labelNumber = dis.readInt()

        return ArrayList<Byte>().apply {
            for (i in 1..labelNumber) {
                add(dis.readUnsignedByte().toByte())
            }
        }
    }
}

fun readMnistTestData(folderPath: String): List<LabeledDigitImage> {
    fun pathOf(p: String) = Paths.get(folderPath, p).toAbsolutePath().toString()
    return readMnistImageFile(pathOf("t10k-images.idx3-ubyte")).zip(
            readMnistLabelFile(pathOf("t10k-labels.idx1-ubyte"))).map {
        val (image, label) = it
        LabeledDigitImage(image, label)
    }
}

fun readMnistTrainingData(folderPath: String): List<LabeledDigitImage> {
    fun pathOf(p: String) = Paths.get(folderPath, p).toAbsolutePath().toString()
    return readMnistImageFile(pathOf("train-images.idx3-ubyte")).zip(
            readMnistLabelFile(pathOf("train-labels.idx1-ubyte"))).map {
        val (image, label) = it
        LabeledDigitImage(image, label)
    }
}

fun showImage(image: BufferedImage, width: Int = 300, height: Int = 300) {
    val canvas = object : Canvas() {
        override fun paint(g: Graphics) {
            g.drawImage(image, 0, 0, width, height, null)
        }
    }.apply {
        size = Dimension(width, height)
        background = Color.BLACK
    }

    val win = JFrame().apply {
        add(canvas)
        pack()
        defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    }

    SwingUtilities.invokeLater {
        win.isVisible = true
    }
}