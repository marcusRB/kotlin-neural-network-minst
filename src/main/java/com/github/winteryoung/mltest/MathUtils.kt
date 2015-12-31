package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector

/**
 * Convert [RealVector] to [RealMatrix].
 */
fun RealVector.toRealMatrix(): RealMatrix {
    val ar = Array(dimension) { i ->
        DoubleArray(1) {
            getEntry(i)
        }
    }
    return MatrixUtils.createRealMatrix(ar)
}

fun RealMatrix.copy(delta: Double): RealMatrix {
    if (delta == 0.0) {
        return copy()
    }
    return copy().apply {
        for (row in 0..rowDimension - 1) {
            for (col in 0..columnDimension - 1) {
                this.setEntry(row, col, getEntry(row, col) + delta)
            }
        }
    }
}

operator fun RealVector.plus(vec: RealVector) = this.add(vec)

operator fun RealVector.plus(d: Double) = this.mapAdd(d)

operator fun Double.plus(vec: RealVector) = vec + this

operator fun RealVector.unaryMinus() = this.map { -it }

operator fun RealVector.minus(vec: RealVector) = this.subtract(vec)

operator fun RealVector.minus(d: Double) = this.mapSubtract(d)

operator fun Double.minus(vec: RealVector) = vec.map { this - it }

operator fun RealVector.times(vec: RealVector) = this.ebeMultiply(vec)

operator fun RealVector.times(v: Double) = this.mapMultiply(v)

operator fun Double.times(vec: RealVector) = vec * this

fun RealVector.pow(v: Double) = this.map { Math.pow(it, v) }

fun Double.pow(vec: RealVector) = vec.map { Math.pow(this, it) }

fun exp(vec: RealVector) = vec.map { Math.exp(it) }

operator fun Double.div(vec: RealVector) = vec.map { this / it }

operator fun RealVector.div(d: Double) = map { it / d }

operator fun RealMatrix.times(vec: RealVector) = this.multiply(vec.toRealMatrix()).getColumnVector(0)

operator fun RealMatrix.minus(m: RealMatrix) = this.subtract(m)

operator fun RealMatrix.times(d: Double) = this.scalarMultiply(d)