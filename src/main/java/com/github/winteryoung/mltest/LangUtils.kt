package com.github.winteryoung.mltest

import java.util.*

fun <T> Collection<T>.split(batchSize: Int): List<List<T>> {
    val result = ArrayList<List<T>>()
    var currentColl = ArrayList<T>()
    this.forEach {  }
    for (item in this) {
        currentColl.add(item)
        if (currentColl.size == batchSize) {
            result.add(currentColl)
            currentColl = ArrayList<T>()
        }
    }
    if (currentColl.isNotEmpty()) {
        result.add(currentColl)
    }
    return result;
}

operator fun <T> List<T>.get(index: Int) = this.getx(index)

fun <T> List<T>.getx(index: Int): T {
    val i = xindex(index)
    return this[i]
}

private fun <T> List<T>.xindex(index: Int): Int {
    val i = if (index < 0) {
        this.size + index
    } else {
        index
    }
    return i
}

fun <T> MutableList<T>.setx(index: Int, value: T) {
    this[xindex(index)] = value
}