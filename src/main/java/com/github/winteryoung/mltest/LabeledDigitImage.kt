package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.ArrayRealVector

/**
 * @author Winter Young
 * @since 2015/12/27
 */
data class LabeledDigitImage(
        val data: Image,
        val label: Byte
) {
    fun toLabeledData(): LabeledData {
        val dataVec = ArrayRealVector(data.data.map { it.toDouble() }.toTypedArray())
        val labelVec = ArrayRealVector(doubleArrayOf( label.toDouble() ))
        return LabeledData(dataVec, labelVec)
    }
}