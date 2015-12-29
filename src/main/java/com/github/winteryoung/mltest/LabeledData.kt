package com.github.winteryoung.mltest

import org.apache.commons.math3.linear.RealVector

/**
 * @author Winter Young
 * @since 2015/12/29
 */
data class LabeledData(
    val data: RealVector,
    val label: RealVector
)