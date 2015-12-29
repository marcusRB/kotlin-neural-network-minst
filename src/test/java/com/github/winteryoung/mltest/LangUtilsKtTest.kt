package com.github.winteryoung.mltest

import org.junit.Assert
import org.junit.Test

/**
 * @author Winter Young
 * @since 2015/12/27
 */
class LangUtilsKtTest {
    @Test
    fun testSplit() {
        val list = listOf(*Array(10) { it + 1 })
        val batches = list.split(3)
        Assert.assertEquals(4, batches.size)
        Assert.assertEquals(listOf(1, 2, 3), batches[0])
        Assert.assertEquals(listOf(10), batches[3])
    }
}