package com.github.winteryoung.mltest

import java.util.concurrent.RecursiveTask

/**
 * @author Winter Young
 * @since 2016/1/3
 */
class DataForkJoinTask(
        private val work: List<LabeledData>,
        private val weights: List<WeightMatrix>,
        private val biases: List<BiasVector>,
        private val trainMiniBatch: (List<LabeledData>) -> Pair<List<WeightMatrix>, List<BiasVector>>
) : RecursiveTask<Pair<List<WeightMatrix>, List<BiasVector>>>() {
    override fun compute(): Pair<List<WeightMatrix>, List<BiasVector>> {
        if (work.size < 5) {
            return trainMiniBatch(work)
        }

        val weightDecsOfBatch = weights.map { it.zero() }
        val biasDecsOfBatch = biases.map { it.zero() }

        val tasks = listOf(
                DataForkJoinTask(work.subList(0, work.size / 2), weights, biases, trainMiniBatch),
                DataForkJoinTask(work.subList(work.size / 2, work.size), weights, biases, trainMiniBatch))
        tasks.forEach { it.fork() }
        tasks.forEach {
            val (wd, bd) = it.join()
            for ((w1, w2) in weightDecsOfBatch.zip(wd)) {
                w1.matrix += w2.matrix
            }
            for ((b1, b2) in biasDecsOfBatch.zip(bd)) {
                b1.matrix += b2.matrix
            }
        }

        return weightDecsOfBatch to biasDecsOfBatch
    }
}