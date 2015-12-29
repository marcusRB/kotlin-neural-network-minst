package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector

/**
 * Convert [RealVector] to [RealMatrix].
 */
fun RealVector.toRealMatrix(): RealMatrix {
    val ar = Array(dimension) {
        DoubleArray(1) {
            getEntry(it)
        }
    }
    return MatrixUtils.createRealMatrix(ar)
}