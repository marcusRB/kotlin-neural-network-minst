package com.github.winteryoung.mltest

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector

/**
 * @author Winter Young
 * @since 2015/12/27
 */
data class BiasVector(
        val size: Int
) {
    val matrix: RealMatrix
    val vector: RealVector
        get() = matrix.getColumnVector(0)

    init {
        val nd = NormalDistribution()
        val m = Array(size) {
            Array(1) {
                nd.sample()
            }.toDoubleArray()
        }
        matrix = MatrixUtils.createRealMatrix(m)
    }

    override fun toString(): String {
        return matrix.toString()
    }
}