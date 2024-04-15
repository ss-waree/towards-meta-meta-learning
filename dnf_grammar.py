import multiprocessing
import time

import math
import numpy as np
import random

import logging
import sys

sys.setrecursionlimit(100)


class Primitive:
    def __init__(self, output_type, input_types, function, name):
        self.output_type = output_type
        self.input_types = input_types
        self.name = name
        self.function = function


class Grammar:
    def __init__(self):
        self.rules = {}
        self.weights = {}

    def add(self, primitive, weight):
        if primitive.output_type not in self.rules:
            self.rules[primitive.output_type] = []
            self.weights[primitive.output_type] = []

        self.rules[primitive.output_type].append(primitive)
        self.weights[primitive.output_type].append(weight)

    def delete(self, output_type):
        if output_type in self.rules:
            del self.rules[output_type]
            del self.weights[output_type]
        else:
            print(f"No rules found for output type '{output_type}'.")

    def __iter__(self):
        return iter(self.rules.items())

    def sample(self, nonterminal):

        if nonterminal in self.rules:
            choices = self.rules[nonterminal]
            weights = self.weights[nonterminal]
        else:
            raise ValueError("No choices available for nonterminal: " + nonterminal)

        choice = random.choices(choices, weights=weights)[0]

        if len(choice.input_types) == 0:
            return choice.function, choice.name

        else:
            arg_functions = []
            arg_names = []

            for input_type in choice.input_types:
                arg_function, arg_name = self.sample(input_type)
                arg_functions.append(arg_function)
                arg_names.append(arg_name)

            new_function = lambda x: choice.function(*[arg_function(x) for arg_function in arg_functions])
            new_name = choice.name % tuple(arg_names)

            return new_function, new_name

    def pretty_print(self):
        for lhs in self.rules:
            for primitive in self.rules[lhs]:
                rhs = primitive.name % tuple(primitive.input_types)
                print(lhs + " -> " + rhs)


class DNFHypothesis:
    def __init__(self, n_features=4, no_true_false_top=True, b=1):

        # Used for determining probability of outlier
        self.b = b
        self.p_outlier = math.exp(-1 * self.b) / (1 + math.exp(-1 * self.b))

        self.grammar = Grammar()

        if no_true_false_top:
            s1 = Primitive("S", ["D_top"], lambda x: x, "∀x l(x) <=> %s")
            self.grammar.add(s1, 1.0)  # Don't have to worry about probability - only one option

            d_top = Primitive("D_top", ["C_top", "D"], lambda x, y: x or y, "(%s or %s)")
            self.grammar.add(d_top, 1.0)  # Don't have to worry about probability - only one option

            c_top = Primitive("C_top", ["P", "C"], lambda x, y: x and y, "(%s and %s)")
            self.grammar.add(c_top, 1.0)  # Don't have to worry about probability - only one option

            d1 = Primitive("D", ["C_top", "D"], lambda x, y: x or y, "(%s or %s)")
            d2 = Primitive("D", [], lambda f: False, "False")

            # Random probabilities for "D" rules
            d_probs = np.random.dirichlet((1, 1))
            p_d1 = d_probs[0]
            p_d2 = d_probs[1]
            self.grammar.add(d1, p_d1)
            self.grammar.add(d2, p_d2)

        else:
            s1 = Primitive("S", ["D"], lambda x: x, "∀x l(x) <=> %s")
            self.grammar.add(s1, 1.0)  # Don't have to worry about probability - only one option

            d1 = Primitive("D", ["C", "D"], lambda x, y: x or y, "(%s or %s)")
            d2 = Primitive("D", [], lambda f: False, "False")

            # Random probabilities for "D" rules
            d_probs = np.random.dirichlet((1, 1))
            p_d1 = d_probs[0]
            p_d2 = d_probs[1]
            self.grammar.add(d1, p_d1)
            self.grammar.add(d2, p_d2)

        c1 = Primitive("C", ["P", "C"], lambda x, y: x and y, "(%s and %s)")
        c2 = Primitive("C", [], lambda f: True, "True")

        # Random probabilities for "C" rules
        c_probs = np.random.dirichlet((1, 1))
        p_c1 = c_probs[0]
        p_c2 = c_probs[1]
        self.grammar.add(c1, p_c1)
        self.grammar.add(c2, p_c2)

        p_probs = np.random.dirichlet([1 for _ in range(n_features)])
        for i in range(n_features):
            p_primitive = Primitive("P", ["F" + str(i + 1)], lambda x: x, "%s")
            self.grammar.add(p_primitive, p_probs[i])

            f_probs = np.random.dirichlet((1, 1))
            f1_primitive = Primitive("F" + str(i + 1), [], lambda f, i=i: f[i] == 1, "f_" + str(i + 1) + "(x) = 1")
            self.grammar.add(f1_primitive, f_probs[0])

            f2_primitive = Primitive("F" + str(i + 1), [], lambda f, i=i: f[i] == 0, "f_" + str(i + 1) + "(x) = 0")
            self.grammar.add(f2_primitive, f_probs[1])

        dataset_created = False
        while not dataset_created:
            # try/except to catch cases with recursion that's too deep
            try:
                self.function, self.name = self.grammar.sample("S")
                # print("FCT", self.function, self.name)
                example_input = [0 for _ in range(n_features)]
                pred = self.function(example_input)
                dataset_created = True
            except:
                pass

    def function_with_outliers(self, inp):

        correct_output = self.function(inp)
        if random.random() < self.p_outlier:
            return not correct_output
        else:
            return correct_output


if __name__ == "__main__":
    my_hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1)
    print(my_hyp.name)

    feature_values = [[0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 1, 1],
                      [0, 1, 0, 0],
                      [0, 1, 0, 1],
                      [0, 1, 1, 0],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1],
                      [1, 0, 1, 0],
                      [1, 0, 1, 1],
                      [1, 1, 0, 0],
                      [1, 1, 0, 1],
                      [1, 1, 1, 0],
                      [1, 1, 1, 1]]

    for features in feature_values:
        print(features, my_hyp.function(features), my_hyp.function_with_outliers(features))

    print("")
    print("Grammar:")
    my_hyp.grammar.pretty_print()

    varied = 0
    total = 0
    for _ in range(1):
        my_hyp = DNFHypothesis()
        all_true = True
        all_false = True
        for features in feature_values:
            output = my_hyp.function(features)
            if output:
                all_false = False
            else:
                all_true = False
        if not all_false and not all_true:
            varied += 1
        total += 1

    print("Varied grammars:", varied, total)
