package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector

/**
 * @author Winter Young
 * @since 2015/12/27
 */
class BiasVector(
        val size: Int,
        valueInitializer: () -> Double = { 0.0 }
) {
    var matrix: RealMatrix
    val vector: RealVector
        get() = matrix.getColumnVector(0)

    init {
        val m = Array(size) {
            Array(1) {
                valueInitializer()
            }.toDoubleArray()
        }
        matrix = MatrixUtils.createRealMatrix(m)
    }

    override fun toString(): String {
        return matrix.toString()
    }

    fun zero() = BiasVector(size)
}