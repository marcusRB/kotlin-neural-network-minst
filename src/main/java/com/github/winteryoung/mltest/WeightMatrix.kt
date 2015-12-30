package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix

/**
 * @author Winter Young
 * @since 2015/12/27
 */
class WeightMatrix(
        val height: Int,
        val width: Int,
        valueInitializer: () -> Double = { 0.0 }
) {
    var matrix: RealMatrix

    init {
        val m = Array(height) {
            Array(width) {
                valueInitializer()
            }.toDoubleArray()
        }
        matrix = MatrixUtils.createRealMatrix(m)
    }

    override fun toString(): String {
        return matrix.toString()
    }

    fun zero() = WeightMatrix(height, width)
}