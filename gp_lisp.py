#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
From data (input-output pairings),
and a set of operators and operands as the only starting point,
write a program that will evolve programmatic solutions,
which take in inputs and generate outputs.

Each program will have 1 numeric input and 1 numeric output.
This is much like regression in our simple case,
though can be generalized much further,
toward arbitrarily large and complex programs.

This assignment is mostly open-ended,
with a couple restrictions:

# DO NOT MODIFY >>>>
Do not edit the sections between these marks below.
# <<<< DO NOT MODIFY
"""

# %%
import random
from typing import TypedDict
from typing import Optional
import copy
import math

# import json

# import math
# import datetime
# import subprocess


# DO NOT MODIFY >>>>
# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
OPERATORS = "+-/*"


class Node:
    """
    Example prefix formula:
    Y = ( * ( + 20 45 ) ( - 56 X ) )
    This is it's tree:
       *
      /  \
    +     -
    / \   / \
    20 45 56  X

    root = Node(
        data="*",
        left=Node(data="+", left=Node("20"), right=Node("45")),
        right=Node(data="-", left=Node("56"), right=Node("X")),
    )
    """

    def __init__(
        self, data: str, left: Optional["Node"] = None, right: Optional["Node"] = None
    ) -> None:
        self.data = data
        self.left = left
        self.right = right


class Individual(TypedDict):
    """Type of each individual to evolve"""

    genome: Node
    fitness: float


Population = list[Individual]


class IOpair(TypedDict):
    """Data type for training and testing data"""

    input1: int
    output1: float


IOdata = list[IOpair]


def print_tree(root: Node, indent: str = "") -> None:
    """
    Pretty-prints the data structure in actual tree form.
    >>> print_tree(root=root, indent="")
    """
    if root.right is not None and root.left is not None:
        print_tree(root=root.right, indent=indent + "    ")
        print(indent, root.data)
        print_tree(root=root.left, indent=indent + "    ")
    else:
        print(indent + root.data)


def parse_expression(source_code: str) -> Node:
    """
    Turns prefix code into a tree data structure.
    >>> clojure_code = "( * ( + 20 45 ) ( - 56 X ) )"
    >>> root = parse_expression(clojure_code)
    """
    source_code = source_code.replace("(", "")
    source_code = source_code.replace(")", "")
    code_arr = source_code.split()
    return _parse_experession(code_arr)


def _parse_experession(code: list[str]) -> Node:
    """
    The back-end helper of parse_expression.
    Not intended for calling directly.
    Assumes code is prefix notation lisp with space delimeters.
    """
    if code[0] in OPERATORS:
        return Node(
            data=code.pop(0),
            left=_parse_experession(code),
            right=_parse_experession(code),
        )
    else:
        return Node(code.pop(0))


def parse_tree_print(root: Node) -> None:
    """
    Stringifies to std-out (print) the tree data structure.
    >>> parse_tree_print(root)
    """
    if root.right is not None and root.left is not None:
        print(f"( {root.data} ", end="")
        parse_tree_print(root.left)
        parse_tree_print(root.right)
        print(") ", end="")
    else:
        # for the case of literal programs... e.g., `4`
        print(f"{root.data} ", end="")


def parse_tree_return(root: Node) -> str:
    """
    Stringifies to the tree data structure, returns string.
    >>> stringified = parse_tree_return(root)
    """
    if root.right is not None and root.left is not None:
        return f"( {root.data} {parse_tree_return(root.left)} {parse_tree_return(root.right)} )"
    else:
        # for the case of literal programs... e.g., `4`
        return root.data


def initialize_individual(genome: str, fitness: float) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as Node, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[Node, int]
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    >>> ind1 = initialize_individual("( + ( * C ( / 9 5 ) ) 32 )", 0)
    """
    return {"genome": parse_expression(genome), "fitness": fitness}


def initialize_data(input1: int, output1: float) -> IOpair:
    """
    For mypy...
    """
    return {"input1": input1, "output1": output1}


def prefix_to_infix(prefix: str) -> str:
    """
    My minimal lisp on python interpreter, lol...
    >>> C = 0
    >>> print(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )"))
    >>> print(eval(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )")))
    """
    prefix = prefix.replace("(", "")
    prefix = prefix.replace(")", "")
    prefix_arr = prefix.split()
    stack = []
    i = len(prefix_arr) - 1
    while i >= 0:
        if prefix_arr[i] not in OPERATORS:
            stack.append(prefix_arr[i])
            i -= 1
        else:
            str = "(" + stack.pop() + prefix_arr[i] + stack.pop() + ")"
            stack.append(str)
            i -= 1
    return stack.pop()


def put_an_x_in_it(formula: str) -> str:
    formula_arr = formula.split()
    while True:
        i = random.randint(0, len(formula_arr) - 1)
        if formula_arr[i] not in OPERATORS:
            formula_arr[i] = "x"
            break
    return " ".join(formula_arr)


def gen_rand_prefix_code(depth_limit: int, rec_depth: int = 0) -> str:
    """
    Generates one small formula,
    from OPERATORS and ints from -100 to 200
    """
    rec_depth += 1
    if rec_depth < depth_limit:
        if random.random() < 0.9:
            return (
                random.choice(OPERATORS)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth)
            )
        else:
            return str(random.randint(-100, 100))
    else:
        return str(random.randint(-100, 100))


# <<<< DO NOT MODIFY


def initialize_pop(pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          random.choice-1, string.ascii_letters-1, initialize_individual-n
    Example doctest:
    """
    population = []
    for i in range(pop_size):
        rand_code = gen_rand_prefix_code(depth_limit=4)
        rand_code = put_an_x_in_it(rand_code)
        new_individual = initialize_individual(rand_code, 0)
        # print(new_individual)
        population.append(new_individual)
    # print(population)
    return population


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          Basic python, random.choice-1, initialize_individual-2
    Example doctest:
    """
    recombined_child_1 = copy.deepcopy(parent1["genome"])
    cnt_child_1 = countNodes(recombined_child_1)
    selected_position_1 = random.choice(range(1, min(2, cnt_child_1) + 1))
    subtree_child_1 = getSubTree(recombined_child_1, selected_position_1, 1)

    recombined_child_2 = copy.deepcopy(parent2["genome"])
    cnt_child_2 = countNodes(recombined_child_2)
    selected_position_2 = random.choice(range(1, min(2, cnt_child_2) + 1))
    subtree_child_2 = getSubTree(recombined_child_2, selected_position_2, 1)

    putSubTree(recombined_child_1, subtree_child_2, selected_position_1, 1)
    putSubTree(recombined_child_2, subtree_child_1, selected_position_2, 1)

    if "x" not in parse_tree_return(recombined_child_1):
        recombined_child_1 = parse_expression(
            put_an_x_in_it(parse_tree_return(recombined_child_1))
        )

    if "x" not in parse_tree_return(recombined_child_2):
        recombined_child_2 = parse_expression(
            put_an_x_in_it(parse_tree_return(recombined_child_2))
        )

    return [
        {"genome": recombined_child_1, "fitness": 0},
        {"genome": recombined_child_2, "fitness": 0},
    ]


def countNodes(node: Node) -> int:
    if node.left == None and node.right == None:
        return 1
    totalNodes: int = 0
    if node.left is not None:
        totalNodes += countNodes(node.left)
    if node.right is not None:
        totalNodes += countNodes(node.right)
    return 1 + totalNodes


def getSubTree(node: Node, pos: int, curr_pos: int) -> Node:
    if curr_pos == pos:
        return copy.deepcopy(node)
    curr_pos += 1
    if node.left is not None:
        return getSubTree(node.left, pos, curr_pos)
    curr_pos += 1
    if node.right is not None:
        return getSubTree(node.right, pos, curr_pos)
    return node


def putSubTree(node: Node, subtreenode: Node, pos: int, curr_pos: int) -> None:
    if curr_pos == pos:
        node.data = subtreenode.data
        node.left = subtreenode.left
        node.right = subtreenode.right
        return
    curr_pos += 1
    if node.left is not None:
        getSubTree(node.left, pos, curr_pos)
    curr_pos += 1
    if node.right is not None:
        getSubTree(node.right, pos, curr_pos)


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          Basic python, random.random~n/2, recombine pair-n
    """
    recombined_population = []
    for i in range(0, len(parents) - 1, 2):
        if recombine_rate > random.random():
            recombined_population.extend(recombine_pair(parents[i], parents[i + 1]))
        else:
            recombined_population.append(parents[i])
            recombined_population.append(parents[i + 1])
    return recombined_population


def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, random,choice-1,
    Example doctest:
    """
    mutatedChild = copy.deepcopy(parent["genome"])
    if random.random() < mutate_rate:
        cnt_child = countNodes(mutatedChild)
        selected_position = random.choice(range(1, min(2, cnt_child) + 1))
        selected_subtree = getSubTree(mutatedChild, selected_position, 1)
        subtree_node_count = countNodes(selected_subtree)
        new_subtree_child = parse_expression(
            gen_rand_prefix_code(depth_limit=round(math.log2(subtree_node_count)))
        )
        putSubTree(mutatedChild, new_subtree_child, selected_position, 1)
        if "x" not in parse_tree_return(mutatedChild):
            mutatedChild = parse_expression(
                put_an_x_in_it(parse_tree_return(mutatedChild))
            )
    return {"genome": mutatedChild, "fitness": 0}


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, mutate_individual-n
    Example doctest:
    """
    mutated_population = []
    for x in children:
        mutated_population.append(mutate_individual(x, mutate_rate))
    return mutated_population


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
    Parameters:     One Individual, data formatted as IOdata
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Notes:          train/test format is like PSB2 (see IOdata above)
    Example doctest:
    >>> evaluate_individual(ind1, io_data)
    """
    fitness = 0
    errors = []
    for sub_eval in io_data:
        eval_string = parse_tree_return(individual["genome"]).replace(
            "x", str(sub_eval["input1"])
        )

        # In clojure, this is really slow with subprocess
        # eval_string = "( float " + eval_string + ")"
        # returnobject = subprocess.run(
        #     ["clojure", "-e", eval_string], capture_output=True
        # )
        # result = float(returnobject.stdout.decode().strip())

        # In python, this is MUCH MUCH faster:
        try:
            y = eval(prefix_to_infix(eval_string))
        except ZeroDivisionError:
            y = math.inf

        errors.append(abs(sub_eval["output1"] - y))
    # Higher errors is bad, and longer strings is bad
    fitness = sum(errors) + len(eval_string.split())
    # Higher fitness is worse
    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          Basic python, evaluate_individual-n
    Example doctest:
    """
    for x in individuals:
        evaluate_individual(x, io_data)


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    ranked_individuals = []
    ranked_individuals = sorted(
        individuals, key=lambda individual: individual["fitness"], reverse=False
    )
    for x in range(len(ranked_individuals)):
        individuals[x] = ranked_individuals[x]


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          Basic python, random.choices-1
    Example doctest:
    """
    fitness_list = []
    selected_individuals = []
    for x in individuals:
        if math.isinf(x["fitness"]):
            continue
        fitness_list.append(x["fitness"])
        selected_individuals.append(x)
    selected_parents = selected_individuals[:number]
    #random.choices(
     #   selected_individuals, weights=fitness_list, k=number
    #)
    return selected_parents


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    """
    return individuals[:pop_size]


def evolve(io_data: IOdata, pop_size: int = 200) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    population = initialize_pop(pop_size=pop_size)
    evaluate_group(individuals=population, io_data=io_data)
    rank_group(individuals=population)
    best_fitness = population[0]["fitness"]
    counter = 0
    while counter < 10000:
        counter += 1
        parents = parent_select(individuals=population, number=80)
        children = recombine_group(parents=parents, recombine_rate=0.8)
        mutants = mutate_group(children=children, mutate_rate=0.02)
        evaluate_group(individuals=mutants, io_data=io_data)
        everyone = population + mutants
        rank_group(individuals=everyone)
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        if best_fitness != population[0]["fitness"]:
            best_fitness = population[0]["fitness"]
            print(
                "Iteration number",
                counter,
                "with best individual",
                parse_tree_return(population[0]["genome"]),
                "fitness",
                population[0]["fitness"],
            )
    return population


# Seed for base grade.
# For the exploratory competition points (last 10),
# comment this one line out if you want, but put it back please.
seed = True

# DO NOT MODIFY >>>>
if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    print(divider)
    print("Number of possible genetic programs: infinite...")
    print("Lower fitness is better.")
    print(divider)

    X = list(range(-10, 110, 10))
    Y = [(x * (9 / 5)) + 32 for x in X]
    # data = [{"input1": x, "output1": y} for x, y in zip(X, Y)]
    # mypy wanted this:
    data = [initialize_data(input1=x, output1=y) for x, y in zip(X, Y)]

    # Correct:
    print("Example of celcius to farenheight:")
    ind1 = initialize_individual("( + ( * x ( / 9 5 ) ) 32 )", 0)
    evaluate_individual(ind1, data)
    print_tree(ind1["genome"])
    print("Fitness", ind1["fitness"])

    # Yours
    train = data[: int(len(data) / 2)]
    test = data[int(len(data) / 2) :]
    population = evolve(train)
    evaluate_individual(population[0], test)
    population[0]["fitness"]

    print("Here is the best program:")
    parse_tree_print(population[0]["genome"])
    print("And it's fitness:")
    print(population[0]["fitness"])
# <<<< DO NOT MODIFY

# %%
