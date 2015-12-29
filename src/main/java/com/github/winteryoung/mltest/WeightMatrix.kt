package com.github.winteryoung.mltest

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector

/**
 * @author Winter Young
 * @since 2015/12/27
 */
class WeightMatrix(
        val height: Int,
        val width: Int
) {
    private val weight: RealMatrix

    init {
        val nd = NormalDistribution()
        val m = Array(height) {
            Array(width) {
                nd.sample()
            }.toDoubleArray()
        }
        weight = MatrixUtils.createRealMatrix(m)
    }

    override fun toString(): String {
        return weight.toString()
    }

    fun multiply(vec: RealVector): RealVector {
        return weight.multiply(vec.toRealMatrix()).getColumnVector(0)
    }
}