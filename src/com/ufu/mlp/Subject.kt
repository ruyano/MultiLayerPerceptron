package com.ufu.mlp

class Subject(
    val attributes: ArrayList<Double> = arrayListOf(),
    val groupIdentification: ArrayList<Double> = arrayListOf()
) {
    override fun toString(): String {
        return "\nSubject(attributes=$attributes, groupIdentification=$groupIdentification)"
    }
}