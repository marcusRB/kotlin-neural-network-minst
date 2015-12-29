package com.github.winteryoung.mltest

import java.awt.image.BufferedImage

/**
 * @author Winter Young
 * @since 2015/12/29
 */
class Image(
        val width: Int,
        val height: Int,
        val data: IntArray
) {
    fun toBufferedImage(): BufferedImage {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        for (h in 0..height - 1) {
            for (w in 0..width - 1) {
                image.setRGB(w, h, data[h * width + w])
            }
        }
        return image;
    }
}