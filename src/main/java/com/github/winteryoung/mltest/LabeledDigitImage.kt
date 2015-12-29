package com.github.winteryoung.mltest

import java.awt.image.BufferedImage

/**
 * @author Winter Young
 * @since 2015/12/27
 */
data class LabeledDigitImage(
        val image: BufferedImage,
        val label: Byte
)