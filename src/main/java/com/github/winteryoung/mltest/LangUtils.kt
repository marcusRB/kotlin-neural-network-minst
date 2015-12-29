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