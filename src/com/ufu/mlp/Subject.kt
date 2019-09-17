package com.ufu.mlp

class Subject(
    val attributes: ArrayList<Double> = arrayListOf(),
    val targetResult: ArrayList<Double> = arrayListOf()
) {
    override fun toString(): String {
        return "\nSubject(attributes=$attributes, groupIdentification=$targetResult)"
    }
}