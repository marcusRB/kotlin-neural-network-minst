package com.github.winteryoung.mltest

import java.util.*
import kotlin.concurrent.getOrSet

/**
 * @author Winter Young
 * @since 2016/1/2
 */
object TraceEnv {
    private val env = ThreadLocal<Stack<String>>()
    private val stack = env.getOrSet({ Stack<String>() })

    fun push(s: String) {
        stack.push(s)
    }

    fun pop(): String {
        return stack.pop()
    }

    override fun toString(): String {
        return stack.joinToString(", ")
    }

    fun <R> use(str: String, action: () -> R): R {
        push(str)
        try {
            return action()
        } finally {
            pop()
        }
    }
}