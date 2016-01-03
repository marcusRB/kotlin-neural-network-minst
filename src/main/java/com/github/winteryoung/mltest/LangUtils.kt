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

fun <T> List<T>.getx(index: Int): T {
    val i = xindex(index)
    return this[i]
}

fun xindex(index: Int, size: Int): Int {
    val i = if (index < 0) {
        size + index
    } else {
        index
    }
    return i
}

private fun <T> List<T>.xindex(index: Int): Int = xindex(index, size)

fun <T> MutableList<T>.setx(index: Int, value: T) {
    this[xindex(index)] = value
}

private fun doubleIsDifferent(d1: Double, d2: Double, delta: Double): Boolean {
    if (java.lang.Double.compare(d1, d2) == 0) {
        return false
    }
    if (Math.abs(d1 - d2) <= delta) {
        return false
    }

    return true
}

fun Double.equals(d: Double, delta: Double) = !doubleIsDifferent(this, d, delta)

fun <T> MutableList<T>.shuffle() {
    Collections.shuffle(this)
}