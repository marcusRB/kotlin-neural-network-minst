package com.github.winteryoung.mltest

import java.util.*

/**
 * @author Winter Young
 * @since 2016/1/2
 */
object TraceEnv {
    private val env = ThreadLocal<Stack<String>>()

    init {
        env.set(Stack<String>())
    }

    fun push(s: String) {
        env.get().push(s)
    }

    fun pop(): String {
        return env.get().pop()
    }

    override fun toString(): String {
        return env.get().joinToString(", ")
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