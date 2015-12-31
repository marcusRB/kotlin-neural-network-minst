package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector

/**
 * @author Winter Young
 * @since 2015/12/27
 */
class BiasVector(var matrix: RealMatrix) {
    constructor(
            size: Int,
            valueInitializer: () -> Double = { 0.0 }
    ) : this(createMatrix(size, valueInitializer))

    val vector: RealVector
        get() = matrix.getColumnVector(0)

    override fun toString(): String {
        return matrix.toString()
    }

    fun copy(delta: Double): BiasVector {
        return BiasVector(matrix.copy(delta))
    }

    fun zero() = BiasVector(MatrixUtils.createRealMatrix(matrix.rowDimension, matrix.columnDimension))

    companion object {
        private fun createMatrix(size: Int, valueInitializer: () -> Double): RealMatrix {
            val m = Array(size) {
                Array(1) {
                    valueInitializer()
                }.toDoubleArray()
            }
            return MatrixUtils.createRealMatrix(m)
        }
    }
}