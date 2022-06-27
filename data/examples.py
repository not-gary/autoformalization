import numpy as np
import string
import argparse
from tokenizer import detokenize_coq, check_proof

from grammars import EvenOdd, Powers, Composites, Poly

parser = argparse.ArgumentParser('Generate Examples')
parser.add_argument('-t', '--type',
                    type=str,
                    default='even-odd',
                    help='Type of dataset to be generated. Must be one of\n\teven-odd (default)\n\tpowers\n\tcomposites\n\tpoly')
parser.add_argument('-p', '--path',
                    type=str,
                    default='examples',
                    help='Filepath to generate dataset to. Default: ./examples')
parser.add_argument('-n', '--num',
                    type=int,
                    default=500,
                    help='Number of examples to generate. Default: 500')
parser.add_argument('-s', '--set',
                    type=str,
                    default='training',
                    help='Subset of data to be generated. Must be one of\n\ttraining (default)\n\tvalidation\n\ttest')
parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Use debug mode. Does not write to files and prints extra information to terminal.')

args = None
args, _ = parser.parse_known_args()

def nat_to_token(n):
    return "<nat:" + str(n) + "> "

def var_to_token(v):
    return "<var:" + v + "> "

def forall(vars, order=None, latex=False):
    n = len(vars)
    if order is not None:
        order = np.delete(order, np.where(order == n))
    else:
        order = np.arange(n)
    
    forall = ""
    for i in range(n - 1):
        forall += "$ " if latex else ""
        forall += var_to_token(vars[order[i]])
        forall += "$ , " if latex else ""
    if n > 1 and latex:
        forall += "and "
    forall += "$ " if latex else ""
    forall += var_to_token(vars[order[n - 1]])

    return forall

def expression(nums, vars, const, order=None, latex=False):
    n = len(nums)
    order = order if order is not None else np.arange(n + 1)
    assert len(order) == n + 1
    times = np.random.choice(["* ", r"\cdot ", r"\times ", ""]) if latex else "* "
    
    expression = ""
    if latex:
        expression += "$ "
        if const is not None:
            for i in range(n + 1):
                if order[i] == n:
                    expression += nat_to_token(const)
                    if i != n:
                        expression += "+ "
                else:
                    nat_first = np.random.binomial(1, 0.5)
                    if nat_first:
                        expression += nat_to_token(nums[order[i]])
                        expression += times
                        expression += var_to_token(vars[order[i]])
                    else:
                        expression += var_to_token(vars[order[i]])
                        expression += times
                        expression += nat_to_token(nums[order[i]])

                    if i != n:
                        expression += "+ "
        else:
            order = order[order != n]
            for i in range(n):
                nat_first = np.random.binomial(1, 0.5)
                if nat_first:
                    expression += nat_to_token(nums[order[i]])
                    expression += times
                    expression += var_to_token(vars[order[i]])
                else:
                    expression += var_to_token(vars[order[i]])
                    expression += times
                    expression += nat_to_token(nums[order[i]])

                if i < n - 1:
                    expression += "+ "
        expression += "$ "
    else:
        if const is not None:
            for i in range(n + 1):
                if order[i] != n:
                    expression += nat_to_token(nums[order[i]])
                    expression += times
                    expression += var_to_token(vars[order[i]])
                    expression += "+ "
            expression += nat_to_token(const)
        else:
            order = order[order != n]
            for i in range(n):
                expression += nat_to_token(nums[order[i]])
                expression += times
                expression += var_to_token(vars[order[i]])
                if i < n - 1:
                    expression += "+ "

    return expression

def even_odd(count):
    grammar = EvenOdd()

    num_options = {
        "training": [2, 3, 5, 7, 9],
        "validation": [2, 3, 4, 5, 7, 8, 9],
        "test": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }

    seed = np.array([
        np.random.choice(num_options[args.set]),    #  0: Number of terms in the expression (including constant).
        np.random.binomial(1, 0.5),                 #  1: Expression is odd.
        np.random.randint(3),                       #  2: Universal quantifier is (0) at begnning of theorem, (1) at end of theorem, (2) absent.
        np.random.randint(3),                       #  3: If [14], then make statement about constant's parity (0) not at all, (1) before claim [14], or (2) after claim [14].
        np.random.binomial(1, 0.5),                 #  4: Make statement(s) about coefficients' parity.
        np.random.binomial(1, 0.5),                 #  5: Make statement(s) about products' parity.
        np.random.binomial(1, 0.5),                 #  6: Make statement that sum without constant is even.
        np.random.binomial(1, 0.5),                 #  7: If [6], then make statement about the addition of even numbers.
        np.random.binomial(1, 0.5),                 #  8: If [6] and not [5], then make statement about multiplication with even numbers.
        np.random.binomial(1, 0.5),                 #  9: Restatement of the theorem.
        np.random.binomial(1, 0.5),                 # 10: If [9], restate at (1) the start or (0) the end of the proof.
        np.random.binomial(1, 0.5),                 # 11: Implicit sublemma to prove sum of products is even.
        np.random.binomial(1, 0.5),                 # 12: If ([4] or [5]) and [7], then state (1) before assertions or (0) after.
        np.random.binomial(1, 0.5),                 # 13: If [4] and [8], then state (1) before assertions or (0) after.
        np.random.binomial(1, 0.5),                 # 14: Make statement about the addition of an even number with an even/odd constant.
        np.random.binomial(1, 0.5),                 # 15: If [14], then state at (1) the start or (0) the end of the proof.
        np.random.binomial(1, 0.5),                 # 16: If not [5], then make statement about multiplication with even numbers.
        np.random.binomial(1, 0.5),                 # 17: If not [5] and [16], then state at (1) the start or (0) the end of the proof.
        np.random.binomial(1, 0.5),                 # 18: If [4] and [5], then pair groups of terms. i.e. 4 is even => 4x is even and 6 is even => 6y is even.
    ])

    coeffs = 2 * np.arange(1, 250)
    np.random.shuffle(coeffs)
    coeffs = coeffs[:seed[0] - 1]

    const = 2 * np.random.randint(250) + seed[1]
    
    vars = np.array(list(string.ascii_letters))
    np.random.shuffle(vars)
    vars = vars[:seed[0] - 1]

    order = np.arange(seed[0])
    np.random.shuffle(order)
    thm_order = np.copy(order)

    # Theorem

    latex = r"\documentclass[12pt]{article} \usepackage{amsthm,amsfonts,amssymb} \newtheorem{theorem}{Theorem} \begin{document} \begin{theorem} "
    coq = "Require Import Arith . Theorem <genP:1> : forall "

    if seed[2] == 0:
        latex += np.random.choice(grammar.given)
        latex += np.random.choice(grammar.any)
        
        phrase = np.random.choice(grammar.natural)
        if phrase in [r"\in \mathbb{N} ", r"\in \mathbb{Z}^+ "]:
            latex += forall(vars, latex=True)
            latex += phrase + "$ "

            coq += forall(vars)
        else:
            latex += phrase
            latex += forall(vars, latex=True) + "$ "

            coq += forall(vars)

        latex += np.random.choice(grammar.expression)
        latex += expression(coeffs, vars, const, order, latex=True)
        latex += np.random.choice(grammar.must_be)
        latex += ["even . ", "odd . "][seed[1]]
    elif seed[2] == 1:
        latex += expression(coeffs, vars, const, order, latex=True)
        latex += np.random.choice(grammar.must_be)
        latex += ["even ", "odd "][seed[1]]
        latex += np.random.choice(grammar.given).lower()
        latex += np.random.choice(grammar.any)
        
        phrase = np.random.choice(grammar.natural)
        if phrase in [r"\in \mathbb{N} ", r"\in \mathbb{Z}^+ "]:
            latex += forall(vars, latex=True)
            latex += phrase + "$ "

            coq += forall(vars)
        else:
            latex += phrase
            latex += forall(vars, latex=True) + "$ "

            coq += forall(vars)

        latex += ". "
    else:
        latex += expression(coeffs, vars, const, order, latex=True)
        latex += np.random.choice(grammar.must_be)
        latex += ["even ", "odd "][seed[1]]
        latex += ". "

        coq += forall(vars, order)

    latex += r"\end{theorem} "

    coq += ": nat , Nat . "
    coq += ["even ", "odd "][seed[1]]
    coq += "( " + expression(coeffs, vars, const, thm_order) + ") "
    coq += "= true . "

    # Proof

    latex += r"\begin{proof} "
    coq += "Proof . intros . "

    sum_of_prods = seed[6] and seed[0] > 2

    restatement_start = seed[9] and seed[10]
    addition_start = seed[14] and seed[15]
    multiplication_start = not sum_of_prods and not seed[5] and seed[16] and seed[17]

    restatement_end = seed[9] and not seed[10]
    addition_end = seed[14] and not seed[15]
    multiplication_end = not sum_of_prods and not seed[5] and seed[16] and not seed[17]

    const_rewrite = False

    if (restatement_start or addition_start or multiplication_start) and not (restatement_end or addition_end or multiplication_end) and seed[3] == 1:
        const_rewrite = True
        coq += "assert ( <genH> : Nat . "
        coq += ["even ", "odd "][seed[1]]
        coq += nat_to_token(const)
        coq += "= true ) . { auto . } "

        latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += "$ " + nat_to_token(const) + "$ "
        latex += np.random.choice(grammar.is_)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]
        latex += ". "

    # Restatement, addition phrase, and multiplication phrase all at start.
    if restatement_start and addition_start and multiplication_start:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        switch = np.random.randint(3)
        if switch == 0: # Rules imply th'm.
            latex += np.random.choice(grammar.since).capitalize()
            latex += np.random.choice(grammar.addition)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]

            latex += np.random.choice(grammar.and_)
            latex += np.random.choice(["", np.random.choice(grammar.since)])
            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.a)
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ", "

            latex += np.random.choice(grammar.then)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
                latex += ". "
        elif switch == 1: # Rules, therefore th'm.
            latex += np.random.choice(grammar.addition).capitalize()
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]
            
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.and_)
            else:
                latex += ". "
                latex += np.random.choice(grammar.further).capitalize()

            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ". "

            latex += np.random.choice(grammar.therefore).capitalize()
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # Th'm, then rules.
            if np.random.binomial(1, 0.5): # We show th'm by rules.
                latex += np.random.choice(grammar.show).capitalize()
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression)
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, const, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += ["even ", "odd "][seed[1]]
                else:
                    latex += np.random.choice(grammar.our)
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
                latex += np.random.choice(grammar.using)
                latex += np.random.choice(grammar.fact)
                latex += np.random.choice(grammar.that)

                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]

                latex += np.random.choice(grammar.and_)
                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
            else: # Th'm. Rules.
                latex += np.random.choice(grammar.show).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "

                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.first).capitalize()
                    latex += np.random.choice(grammar.notice)
                else:
                    latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]

                if np.random.binomial(1, 0.5):
                    latex += ", "
                    latex += np.random.choice(grammar.and_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.next)
                    latex += np.random.choice(grammar.notice)
                else:
                    latex += ". "
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.next).capitalize()
                        latex += np.random.choice(grammar.notice)
                    else:
                        latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
    # Restatement and addition phrase at start.
    elif restatement_start and addition_start:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            if not sum_of_prods:
                coq += "rewrite <- Nat . negb_even . "
                coq += "repeat rewrite Nat . even_add . "
        else:
            if not sum_of_prods:
                coq += "repeat "
            coq += "rewrite Nat . even_add . "
        
        switch = np.random.randint(3)
        if switch == 0: # Rules imply th'm.
            latex += np.random.choice(grammar.since).capitalize()
            latex += np.random.choice(grammar.addition)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]
            latex += ", "

            latex += np.random.choice(grammar.then)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
                latex += ". "
        elif switch == 1: # Rules, therefore th'm.
            latex += np.random.choice(grammar.addition).capitalize()
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]
            latex += ". "

            latex += np.random.choice(grammar.therefore).capitalize()
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # Th'm, then rules.
            if np.random.binomial(1, 0.5): # We show th'm by rules.
                latex += np.random.choice(grammar.show).capitalize()
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression)
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, const, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += ["even ", "odd "][seed[1]]
                else:
                    latex += np.random.choice(grammar.our)
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
                latex += np.random.choice(grammar.using)
                latex += np.random.choice(grammar.fact)
                latex += np.random.choice(grammar.that)

                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else: # Th'm. Rules.
                latex += np.random.choice(grammar.show).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "

                latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]
                latex += ". "
    # Restatement and multiplication phrase at start.
    elif restatement_start and multiplication_start:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        switch = np.random.randint(3)
        if switch == 0: # Rules imply th'm.
            latex += np.random.choice(grammar.since).capitalize()
            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.a)
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ", "

            latex += np.random.choice(grammar.then)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
                latex += ". "
        elif switch == 1: # Rules, therefore th'm.
            latex += np.random.choice(grammar.multiplication).capitalize()
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ". "

            latex += np.random.choice(grammar.therefore).capitalize()
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # Th'm, then rules.
            if np.random.binomial(1, 0.5): # We show th'm by rules.
                latex += np.random.choice(grammar.show).capitalize()
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression)
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, const, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += ["even ", "odd "][seed[1]]
                else:
                    latex += np.random.choice(grammar.our)
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
                latex += np.random.choice(grammar.using)
                latex += np.random.choice(grammar.fact)
                latex += np.random.choice(grammar.that)

                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
            else: # Th'm. Rules.
                latex += np.random.choice(grammar.show).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "

                latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
    # Addition and multiplication phrases at start.
    elif addition_start and multiplication_start:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.first).capitalize()
            latex += np.random.choice(grammar.notice)
        else:
            latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += np.random.choice(grammar.addition)
        latex += np.random.choice(grammar.of)
        if np.random.binomial(1, 0.5):
            latex += "even "
            latex += np.random.choice(grammar.numbers)
            latex += np.random.choice(grammar.together)
            if seed[1] or np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
        else:
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]

        if np.random.binomial(1, 0.5):
            latex += ", "
            latex += np.random.choice(grammar.and_)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.next)
            latex += np.random.choice(grammar.notice)
        else:
            latex += ". "
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.next).capitalize()
                latex += np.random.choice(grammar.notice)
            else:
                latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += np.random.choice(grammar.multiplication)
        latex += np.random.choice(grammar.of)
        latex += np.random.choice(grammar.an)
        latex += "even "
        latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.with_)
        latex += np.random.choice(grammar.a)
        latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
        else:
            latex += "even "
        latex += ". "
    # Restatement at start.
    elif restatement_start:
        latex += np.random.choice(grammar.show).capitalize()
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.expression)
            np.random.shuffle(order)
            latex += expression(coeffs, vars, const, order, latex=True)
            latex += np.random.choice(grammar.must_be)
            latex += ["even ", "odd "][seed[1]]
        else:
            latex += np.random.choice(grammar.our)
            latex += np.random.choice(grammar.theorem)
            latex += np.random.choice(grammar.holds)
        latex += ". "
    # Addition phrase at start.
    elif addition_start:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            if not sum_of_prods:
                coq += "rewrite <- Nat . negb_even . "
                coq += "repeat rewrite Nat . even_add . "
        else:
            if not sum_of_prods:
                coq += "repeat "
            coq += "rewrite Nat . even_add . "

        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.notice).capitalize()
            latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.addition)
        else:
            latex += np.random.choice(grammar.addition).capitalize()
        latex += np.random.choice(grammar.of)
        if np.random.binomial(1, 0.5):
            latex += "even "
            latex += np.random.choice(grammar.numbers)
            latex += np.random.choice(grammar.together)
            if seed[1] or np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
        else:
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]
        latex += ". "
    # Multiplication phrase at start.
    elif multiplication_start:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.notice).capitalize()
            latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.multiplication)
        else:
            latex += np.random.choice(grammar.multiplication).capitalize()
        latex += np.random.choice(grammar.of)
        if np.random.binomial(1, 0.5):
            latex += "even "
            latex += np.random.choice(grammar.numbers)
        else:
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.a)
            latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
        else:
            latex += "even "
        latex += ". "

    if (restatement_start or addition_start or multiplication_start) and not (restatement_end or addition_end or multiplication_end) and seed[3] == 2:
        const_rewrite = True
        coq += "assert ( <genH> : Nat . "
        coq += ["even ", "odd "][seed[1]]
        coq += nat_to_token(const)
        coq += "= true ) . { auto . } "

        latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += "$ " + nat_to_token(const) + "$ "
        latex += np.random.choice(grammar.is_)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]
        latex += ". "

    if sum_of_prods:
        switch = np.random.randint(5)
        if switch == 0:
            latex += np.random.choice(grammar.show).capitalize()
            latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.expression)
        elif switch == 1:
            latex += np.random.choice(grammar.notice).capitalize()
            latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.expression)
        elif switch == 2:
            latex += np.random.choice(grammar.expression).capitalize()
        elif switch == 3:
            latex += np.random.choice(grammar.use).capitalize()
            latex += np.random.choice(grammar.fact)
            latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.expression)
        elif switch == 4:
            latex += np.random.choice(grammar.our).capitalize()
            latex += np.random.choice(grammar.theorem)
            latex += np.random.choice(grammar.holds)
            latex += np.random.choice(grammar.since)

        if switch < 4 and np.random.binomial(1, 0.5):
            np.random.shuffle(order)
            latex += expression(coeffs, vars, const, order, latex=True)
            latex += np.random.choice(grammar.is_)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.since)
            latex += np.random.choice(grammar.expression)

        np.random.shuffle(order)
        latex += expression(coeffs, vars, None, order, latex=True)
        latex += np.random.choice(grammar.is_)
        latex += "even "

        early_add = (seed[4] or seed[5]) and seed[7] and seed[12]
        if early_add:
            if np.random.binomial(1, 0.5):
                latex += ", "
                latex += np.random.choice(grammar.since)
            else:
                latex += ". "
                latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.addition)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "

        early_mul = seed[4] and (not seed[5] and seed[8]) and seed[13]
        if early_mul:
            if not early_add and np.random.binomial(1, 0.5):
                latex += ", "
                latex += np.random.choice(grammar.since)
            elif early_add and np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.and_)
                latex += np.random.choice(grammar.fact)
                latex += np.random.choice(grammar.that)
            else:
                latex += ". "
                latex += np.random.choice(grammar.notice).capitalize()
                if early_add:
                    latex += np.random.choice(grammar.also)
                latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
        latex += ". "

        if seed[4] and seed[5] and seed[18]:
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            if np.random.binomial(1, 0.5): # Individual pairs.
                for i in range(seed[0]):
                    if order[i] == seed[0] - 1:
                        continue

                    coq += "assert ( <genH> : Nat . even "
                    coq += nat_to_token(coeffs[order[i]])
                    coq += "= true ) . { auto . } "
                    coq += "assert ( <genH> : Nat . even ( "
                    coq += nat_to_token(coeffs[order[i]])
                    coq += "* "
                    coq += var_to_token(vars[order[i]])
                    coq += ") = true ) . { rewrite Nat . even_mul . rewrite <genH> . auto . } "

                    forward = np.random.binomial(1, 0.5)
                    if i > 0 and np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.further).capitalize()
                        latex += ", "
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice)
                            latex += np.random.choice(grammar.that)
                        
                        if forward:
                            latex += np.random.choice(grammar.coefficient)
                        else:
                            latex += np.random.choice(grammar.product)
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice).capitalize()
                            latex += np.random.choice(grammar.that)
                            if forward:
                                latex += np.random.choice(grammar.coefficient)
                            else:
                                latex += np.random.choice(grammar.product)
                        elif forward:
                            latex += np.random.choice(grammar.coefficient).capitalize()
                        else:
                            latex += np.random.choice(grammar.product).capitalize()

                    latex += "$ "
                    if forward:
                        latex += nat_to_token(coeffs[order[i]])
                    elif np.random.binomial(1, 0.5):
                        latex += nat_to_token(coeffs[order[i]])
                        latex += times
                        latex += var_to_token(vars[order[i]])
                    else:
                        latex += var_to_token(vars[order[i]])
                        latex += times
                        latex += nat_to_token(coeffs[order[i]])
                    latex += "$ "

                    latex += np.random.choice(grammar.is_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "

                    if forward:
                        latex += ". "
                        latex += np.random.choice(grammar.therefore).capitalize()
                        latex += np.random.choice(grammar.product)
                        latex += "$ "
                        if np.random.binomial(1, 0.5):
                            latex += nat_to_token(coeffs[order[i]])
                            latex += times
                            latex += var_to_token(vars[order[i]])
                        else:
                            latex += var_to_token(vars[order[i]])
                            latex += times
                            latex += nat_to_token(coeffs[order[i]])
                        latex += "$ "
                        latex += np.random.choice(grammar.must_be)
                        latex += "even "
                        latex += np.random.choice(grammar.also)
                        latex += ". "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.since)
                        else:
                            latex += ". "
                            latex += np.random.choice(grammar.this).capitalize()
                            latex += np.random.choice(grammar.holds)
                            latex += np.random.choice(grammar.since)

                        if np.random.binomial(1, 0.5): # State fact about multiplication.
                            latex += np.random.choice(grammar.multiplication)
                            latex += np.random.choice(grammar.of)
                            latex += np.random.choice(grammar.an)
                            latex += "even "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.with_)
                            latex += np.random.choice(grammar.a)
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.must_be)
                            if np.random.binomial(1, 0.5):
                                latex += np.random.choice(grammar.itself)
                                latex += np.random.choice(grammar.an)
                                latex += "even "
                                latex += np.random.choice(grammar.number)
                            else:
                                latex += "even "

                            switch = np.random.randint(3)
                            if switch == 0:
                                latex += np.random.choice(grammar.and_)
                                latex += np.random.choice(grammar.coefficient)
                                latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                                latex += np.random.choice(grammar.is_)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.itself)
                                    latex += np.random.choice(grammar.an)
                                    latex += "even "
                                    latex += np.random.choice(grammar.number)
                                else:
                                    latex += "even "
                                latex += ". "
                            else:
                                if switch == 1:
                                    latex += ", "
                                    latex += np.random.choice(grammar.which)
                                else:
                                    latex += ". "
                                    latex += np.random.choice(grammar.this).capitalize()
                                latex += np.random.choice(grammar.holds)
                                latex += np.random.choice(grammar.since)
                                latex += np.random.choice(grammar.coefficient)
                                latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                                latex += np.random.choice(grammar.is_)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.itself)
                                    latex += np.random.choice(grammar.an)
                                    latex += "even "
                                    latex += np.random.choice(grammar.number)
                                else:
                                    latex += "even "
                                latex += ". "
                        else:
                            latex += np.random.choice(grammar.coefficient)
                            latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                            latex += np.random.choice(grammar.is_)
                            if np.random.binomial(1, 0.5):
                                latex += np.random.choice(grammar.itself)
                                latex += np.random.choice(grammar.an)
                                latex += "even "
                                latex += np.random.choice(grammar.number)
                            else:
                                latex += "even "
                            latex += ". "
            else: # Random length pairs.
                low = 0
                coeffs_order = np.arange(seed[0] - 1)
                np.random.shuffle(coeffs_order)
                while low < seed[0] - 1:
                    high = np.random.randint(low + 1, seed[0] - 1) if low + 1 < seed[0] - 1 else low + 1

                    for i in range(low, high):
                        coq += "assert ( <genH> : Nat . even "
                        coq += nat_to_token(coeffs[coeffs_order[i]])
                        coq += "= true ) . { auto . } "
                    
                    for i in range(low, high):
                        coq += "assert ( <genH> : Nat . even ( "
                        coq += nat_to_token(coeffs[coeffs_order[i]])
                        coq += "* "
                        coq += var_to_token(vars[coeffs_order[i]])
                        coq += ") = true ) . { rewrite Nat . even_mul . rewrite <genH> . auto . } "

                    forward = np.random.binomial(1, 0.5)
                    if low > 0 and np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.further).capitalize()
                        latex += ", "
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice)
                            latex += np.random.choice(grammar.that)
                        
                        if forward and high - low > 1:
                            latex += np.random.choice(grammar.coefficients)
                        elif forward:
                            latex += np.random.choice(grammar.coefficient)
                        elif high - low > 1:
                            latex += np.random.choice(grammar.products)
                        else:
                            latex += np.random.choice(grammar.product)
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice).capitalize()
                            latex += np.random.choice(grammar.that)
                            if forward and high - low > 1:
                                latex += np.random.choice(grammar.coefficients)
                            elif forward:
                                latex += np.random.choice(grammar.coefficient)
                            elif high - low > 1:
                                latex += np.random.choice(grammar.products)
                            else:
                                latex += np.random.choice(grammar.product)
                        elif forward and high - low > 1:
                            latex += np.random.choice(grammar.coefficients).capitalize()
                        elif forward:
                            latex += np.random.choice(grammar.coefficient).capitalize()
                        elif high - low > 1:
                            latex += np.random.choice(grammar.products).capitalize()
                        else:
                            latex += np.random.choice(grammar.product).capitalize()

                    for i in range(low, high):
                        if high - low > 1 and i == high - 1:
                            latex += np.random.choice(grammar.and_)
                        latex += "$ "
                        if forward:
                            latex += nat_to_token(coeffs[coeffs_order[i]])
                        elif np.random.binomial(1, 0.5):
                            latex += nat_to_token(coeffs[coeffs_order[i]])
                            latex += times
                            latex += var_to_token(vars[coeffs_order[i]])
                        else:
                            latex += var_to_token(vars[coeffs_order[i]])
                            latex += times
                            latex += nat_to_token(coeffs[coeffs_order[i]])
                        latex += "$ "
                        if high - low > 2 and i < high - 1:
                            latex += ", "

                    if high - low > 1:
                        latex += np.random.choice(grammar.are)
                        if np.random.binomial(1, 0.5):
                            latex += "even "
                            latex += np.random.choice(grammar.numbers)
                        else:
                            latex += "even "
                    else:
                        latex += np.random.choice(grammar.is_)
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.an)
                            latex += "even "
                            latex += np.random.choice(grammar.number)
                        else:
                            latex += "even "

                    if forward:
                        latex += ". "
                        latex += np.random.choice(grammar.therefore).capitalize()
                        if high - low > 1:
                            latex += np.random.choice(grammar.products)
                        else:
                            latex += np.random.choice(grammar.product)
                        for i in range(low, high):
                            if high - low > 1 and i == high - 1:
                                latex += np.random.choice(grammar.and_)
                            latex += "$ "
                            if np.random.binomial(1, 0.5):
                                latex += nat_to_token(coeffs[coeffs_order[i]])
                                latex += times
                                latex += var_to_token(vars[coeffs_order[i]])
                            else:
                                latex += var_to_token(vars[coeffs_order[i]])
                                latex += times
                                latex += nat_to_token(coeffs[coeffs_order[i]])
                            latex += "$ "
                            if high - low > 2 and i < high - 1:
                                latex += ", "
                        latex += np.random.choice(grammar.must_be)
                        latex += "even "
                        latex += np.random.choice(grammar.also)
                        latex += ". "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.since)
                        else:
                            latex += ". "
                            latex += np.random.choice(grammar.this).capitalize()
                            latex += np.random.choice(grammar.holds)
                            latex += np.random.choice(grammar.since)

                        if np.random.binomial(1, 0.5): # State fact about multiplication.
                            latex += np.random.choice(grammar.multiplication)
                            latex += np.random.choice(grammar.of)
                            latex += np.random.choice(grammar.an)
                            latex += "even "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.with_)
                            latex += np.random.choice(grammar.a)
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.must_be)
                            if np.random.binomial(1, 0.5):
                                latex += np.random.choice(grammar.itself)
                                latex += np.random.choice(grammar.an)
                                latex += "even "
                                latex += np.random.choice(grammar.number)
                            else:
                                latex += "even "

                            switch = np.random.randint(3)
                            if switch == 0:
                                latex += np.random.choice(grammar.and_)
                                if high - low > 1:
                                    latex += np.random.choice(grammar.coefficients)
                                else:
                                    latex += np.random.choice(grammar.coefficient)

                                for i in range(low, high):
                                    if high - low > 1 and i == high - 1:
                                        latex += np.random.choice(grammar.and_)
                                    latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                                    if high - low > 2 and i < high - 1:
                                        latex += ", "

                                if high - low > 1:
                                    latex += np.random.choice(grammar.are)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.themselves)
                                        latex += "even "
                                        latex += np.random.choice(grammar.numbers)
                                    else:
                                        latex += "even "
                                else:
                                    latex += np.random.choice(grammar.is_)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.itself)
                                        latex += np.random.choice(grammar.an)
                                        latex += "even "
                                        latex += np.random.choice(grammar.number)
                                    else:
                                        latex += "even "
                                latex += ". "
                            else:
                                if switch == 1:
                                    latex += ", "
                                    latex += np.random.choice(grammar.which)
                                else:
                                    latex += ". "
                                    latex += np.random.choice(grammar.this).capitalize()
                                latex += np.random.choice(grammar.holds)
                                latex += np.random.choice(grammar.since)
                                if high - low > 1:
                                    latex += np.random.choice(grammar.coefficients)
                                else:
                                    latex += np.random.choice(grammar.coefficient)
                                
                                for i in range(low, high):
                                    if high - low > 1 and i == high - 1:
                                        latex += np.random.choice(grammar.and_)
                                    latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                                    if high - low > 2 and i < high - 1:
                                        latex += ", "

                                if high - low > 1:
                                    latex += np.random.choice(grammar.are)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.themselves)
                                        latex += "even "
                                        latex += np.random.choice(grammar.numbers)
                                    else:
                                        "even "
                                else:
                                    latex += np.random.choice(grammar.is_)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.itself)
                                        latex += np.random.choice(grammar.an)
                                        latex += "even "
                                        latex += np.random.choice(grammar.number)
                                    else:
                                        latex += "even "
                                latex += ". "
                        else:
                            if high - low > 1:
                                latex += np.random.choice(grammar.coefficients)
                            else:
                                latex += np.random.choice(grammar.coefficient)
                            
                            for i in range(low, high):
                                if high - low > 1 and i == high - 1:
                                    latex += np.random.choice(grammar.and_)
                                latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                                if high - low > 2 and i < high - 1:
                                    latex += ", "

                            if high - low > 1:
                                latex += np.random.choice(grammar.are)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.themselves)
                                    latex += "even "
                                    latex += np.random.choice(grammar.numbers)
                                else:
                                    latex += "even "
                            else:
                                latex += np.random.choice(grammar.is_)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.itself)
                                    latex += np.random.choice(grammar.an)
                                    latex += "even "
                                    latex += np.random.choice(grammar.number)
                                else:
                                    latex += "even "
                            latex += ". "
                    low = high
        elif seed[4] and seed[5]:
            times = np.random.choice(grammar.times)

            for i in range(seed[0]):
                if order[i] == seed[0] - 1:
                    continue
                coq += "assert ( <genH> : Nat . even "
                coq += nat_to_token(coeffs[order[i]])
                coq += "= true ) . { auto . } "

            for i in range(seed[0]):
                if order[i] == seed[0] - 1:
                    continue
                coq += "assert ( <genH> : Nat . even ( "
                coq += nat_to_token(coeffs[order[i]])
                coq += "* "
                coq += var_to_token(vars[order[i]])
                coq += ") = true ) . { rewrite Nat . even_mul . rewrite <genH> . auto . } "

            forward = np.random.binomial(1, 0.5)
            if i > 0 and np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.further).capitalize()
                latex += ", "
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.notice)
                    latex += np.random.choice(grammar.that)
                
                if forward:
                    latex += np.random.choice(grammar.coefficients)
                else:
                    latex += np.random.choice(grammar.products)
            else:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.notice).capitalize()
                    latex += np.random.choice(grammar.that)
                    if forward:
                        latex += np.random.choice(grammar.coefficients)
                    else:
                        latex += np.random.choice(grammar.products)
                elif forward:
                    latex += np.random.choice(grammar.coefficients).capitalize()
                else:
                    latex += np.random.choice(grammar.products).capitalize()

            for i in range(seed[0]):
                if order[i] == seed[0] - 1:
                    continue

                if seed[0] > 2 and i == seed[0] - 1:
                    latex += np.random.choice(grammar.and_)
                latex += "$ "
                if forward:
                    latex += nat_to_token(coeffs[order[i]])
                elif np.random.binomial(1, 0.5):
                    latex += nat_to_token(coeffs[order[i]])
                    latex += times
                    latex += var_to_token(vars[order[i]])
                else:
                    latex += var_to_token(vars[order[i]])
                    latex += times
                    latex += nat_to_token(coeffs[order[i]])
                latex += "$ "
                if seed[0] > 3 and i < seed[0] - 1:
                    latex += ", "

            latex += np.random.choice(grammar.are)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += "even "

            if forward:
                latex += ". "
                latex += np.random.choice(grammar.therefore).capitalize()
                latex += np.random.choice(grammar.products)

                for i in range(seed[0]):
                    if order[i] == seed[0] - 1:
                        continue

                    if seed[0] > 2 and i == seed[0] - 1:
                        latex += np.random.choice(grammar.and_)
                    latex += "$ "
                    if np.random.binomial(1, 0.5):
                        latex += nat_to_token(coeffs[order[i]])
                        latex += times
                        latex += var_to_token(vars[order[i]])
                    else:
                        latex += var_to_token(vars[order[i]])
                        latex += times
                        latex += nat_to_token(coeffs[order[i]])
                    latex += "$ "
                    if seed[0] > 3 and i < seed[0] - 1:
                        latex += ", "

                latex += np.random.choice(grammar.must_be)
                latex += "even "
                latex += np.random.choice(grammar.also)
                latex += ". "
            else:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.since)
                else:
                    latex += ". "
                    latex += np.random.choice(grammar.this).capitalize()
                    latex += np.random.choice(grammar.holds)
                    latex += np.random.choice(grammar.since)

                if np.random.binomial(1, 0.5): # State fact about multiplication.
                    latex += np.random.choice(grammar.multiplication)
                    latex += np.random.choice(grammar.of)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.a)
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.must_be)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.itself)
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "

                    switch = np.random.randint(3)
                    if switch == 0:
                        latex += np.random.choice(grammar.and_)
                        latex += np.random.choice(grammar.coefficients)

                        for i in range(seed[0]):
                            if order[i] == seed[0] - 1:
                                continue

                            if seed[0] > 2 and i == seed[0] - 1:
                                latex += np.random.choice(grammar.and_)
                            latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                            if seed[0] > 3 and i < seed[0] - 1:
                                latex += ", "

                        latex += np.random.choice(grammar.are)
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.themselves)
                            latex += "even "
                            latex += np.random.choice(grammar.numbers)
                        else:
                            latex += "even "
                        latex += ". "
                    else:
                        if switch == 1:
                            latex += ", "
                            latex += np.random.choice(grammar.which)
                        else:
                            latex += ". "
                            latex += np.random.choice(grammar.this).capitalize()
                        latex += np.random.choice(grammar.holds)
                        latex += np.random.choice(grammar.since)
                        latex += np.random.choice(grammar.coefficients)

                        for i in range(seed[0]):
                            if order[i] == seed[0] - 1:
                                continue

                            if seed[0] > 2 and i == seed[0] - 1:
                                latex += np.random.choice(grammar.and_)
                            latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                            if seed[0] > 3 and i < seed[0] - 1:
                                latex += ", "

                        latex += np.random.choice(grammar.are)
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.themselves)
                            latex += "even "
                            latex += np.random.choice(grammar.numbers)
                        else:
                            latex += "even "
                        latex += ". "
                else:
                    latex += np.random.choice(grammar.coefficients)

                    for i in range(seed[0]):
                        if order[i] == seed[0] - 1:
                            continue
                        
                        if seed[0] > 2 and i == seed[0] - 1:
                            latex += np.random.choice(grammar.and_)
                        latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                        if seed[0] > 3 and i < seed[0] - 1:
                            latex += ", "

                    latex += np.random.choice(grammar.are)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.themselves)
                        latex += "even "
                        latex += np.random.choice(grammar.numbers)
                    else:
                        latex += "even "
                    latex += ". "
        elif seed[4]:
            low = 0
            coeffs_order = np.arange(seed[0] - 1)
            np.random.shuffle(coeffs_order)
            while low < seed[0] - 1:
                high = np.random.randint(low + 1, seed[0] - 1) if low + 1 < seed[0] - 1 else low + 1

                for i in range(low, high):
                    coq += "assert ( <genH> : Nat . even "
                    coq += nat_to_token(coeffs[coeffs_order[i]])
                    coq += "= true ) . { auto . } "

                if low > 0 and np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += ", "
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice)
                        latex += np.random.choice(grammar.that)
                    
                    if high - low > 1:
                        latex += np.random.choice(grammar.coefficients)
                    else:
                        latex += np.random.choice(grammar.coefficient)
                else:
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice).capitalize()
                        latex += np.random.choice(grammar.that)
                        if high - low > 1:
                            latex += np.random.choice(grammar.coefficients)
                        else:
                            latex += np.random.choice(grammar.coefficient)
                    elif high - low > 1:
                        latex += np.random.choice(grammar.coefficients).capitalize()
                    else:
                        latex += np.random.choice(grammar.coefficient).capitalize()

                for i in range(low, high):
                    if high - low > 1 and i == high - 1:
                        latex += np.random.choice(grammar.and_)
                    latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                    if high - low > 2 and i < high - 1:
                        latex += ", "

                if high - low > 1:
                    latex += np.random.choice(grammar.are)
                    if np.random.binomial(1, 0.5):
                        latex += "even "
                        latex += np.random.choice(grammar.numbers)
                    else:
                        latex += "even "
                else:
                    latex += np.random.choice(grammar.is_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "
                low = high
        elif seed[5]:
            times = np.random.choice(grammar.times)
            
            low = 0
            coeffs_order = np.arange(seed[0] - 1)
            np.random.shuffle(coeffs_order)
            while low < seed[0] - 1:
                high = np.random.randint(low + 1, seed[0] - 1) if low + 1 < seed[0] - 1 else low + 1

                for i in range(low, high):
                    coq += "assert ( <genH> : Nat . even ( "
                    coq += nat_to_token(coeffs[coeffs_order[i]])
                    coq += "* "
                    coq += var_to_token(vars[coeffs_order[i]])
                    coq += ") = true ) . { rewrite Nat . even_mul . auto . } "

                if low > 0 and np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += ", "
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice)
                        latex += np.random.choice(grammar.that)
                    
                    if high - low > 1:
                        latex += np.random.choice(grammar.products)
                    else:
                        latex += np.random.choice(grammar.product)
                else:
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice).capitalize()
                        latex += np.random.choice(grammar.that)
                        if high - low > 1:
                            latex += np.random.choice(grammar.products)
                        else:
                            latex += np.random.choice(grammar.product)
                    elif high - low > 1:
                        latex += np.random.choice(grammar.products).capitalize()
                    else:
                        latex += np.random.choice(grammar.product).capitalize()

                for i in range(low, high):
                    if high - low > 1 and i == high - 1:
                        latex += np.random.choice(grammar.and_)
                    latex += "$ "
                    if np.random.binomial(1, 0.5):
                        latex += nat_to_token(coeffs[coeffs_order[i]])
                        latex += times
                        latex += var_to_token(vars[coeffs_order[i]])
                    else:
                        latex += var_to_token(vars[coeffs_order[i]])
                        latex += times
                        latex += nat_to_token(coeffs[coeffs_order[i]])
                    latex += "$ "
                    if high - low > 2 and i < high - 1:
                        latex += ", "

                if high - low > 1:
                    latex += np.random.choice(grammar.are)
                    if np.random.binomial(1, 0.5):
                        latex += "even "
                        latex += np.random.choice(grammar.numbers)
                    else:
                        latex += "even "
                else:
                    latex += np.random.choice(grammar.is_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "

                if np.random.binomial(1, 0.5):
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.since)
                    else:
                        latex += ". "
                        latex += np.random.choice(grammar.this).capitalize()
                        latex += np.random.choice(grammar.holds)
                        latex += np.random.choice(grammar.since)

                    latex += np.random.choice(grammar.multiplication)
                    latex += np.random.choice(grammar.of)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.a)
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.must_be)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.itself)
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "
                latex += ". "
                low = high

        forward = np.random.binomial(1, 0.5)
        if not early_add and seed[7]:
            if forward:
                latex += np.random.choice(grammar.since).capitalize()
                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ", "

                if not seed[8]:
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.expression)
                        np.random.shuffle(order)
                        latex += expression(coeffs, vars, None, order, latex=True)
                        latex += np.random.choice(grammar.must_be)
                        latex += np.random.choice(grammar.itself)
                        latex += "even "
                    else:
                        latex += np.random.choice(grammar.our)
                        latex += np.random.choice(grammar.theorem)
                        latex += np.random.choice(grammar.holds)
                    latex += ". "
            else:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression).capitalize()
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, None, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += np.random.choice(grammar.itself)
                    latex += "even "
                else:
                    latex += np.random.choice(grammar.our).capitalize()
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)

                latex += np.random.choice(grammar.since)
                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "

                if not seed[8] or early_mul:
                    latex += ". "

        if not early_mul and seed[8]:
            if seed[7]:
                latex += np.random.choice(grammar.and_)
                latex += np.random.choice(grammar.since)
            elif forward:
                latex += np.random.choice(grammar.since).capitalize()
            else:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression).capitalize()
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, None, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += np.random.choice(grammar.itself)
                    latex += "even "
                else:
                    latex += np.random.choice(grammar.our).capitalize()
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
                latex += np.random.choice(grammar.since)

            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.a)
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "

            if (not seed[7] or early_add) and forward:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression)
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, None, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += np.random.choice(grammar.itself)
                    latex += "even "
                else:
                    latex += np.random.choice(grammar.our)
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
            latex += ". "

        coq += "assert ( <genH> : Nat . even ( "
        coq += expression(coeffs, vars, None, thm_order)
        coq += ") = true ) . { "
        coq += "repeat rewrite Nat . even_add . "
        if seed[5]:
            for i in range(seed[0] - 1):
                coq += "rewrite <genH> . "
        elif seed[4]:
            coq += "repeat rewrite Nat . even_mul . "
            for i in range(seed[0] - 1):
                coq += "rewrite <genH> . "
        else:
            coq += "repeat rewrite Nat . even_mul . "
        coq += "auto . } "
    else:
        if seed[4] and seed[5] and seed[18]:
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            if np.random.binomial(1, 0.5): # Individual pairs.
                for i in range(seed[0]):
                    if order[i] == seed[0] - 1:
                        continue

                    coq += "assert ( <genH> : Nat . even "
                    coq += nat_to_token(coeffs[order[i]])
                    coq += "= true ) . { auto . } "
                    coq += "assert ( <genH> : Nat . even ( "
                    coq += nat_to_token(coeffs[order[i]])
                    coq += "* "
                    coq += var_to_token(vars[order[i]])
                    coq += ") = true ) . { rewrite Nat . even_mul . rewrite <genH> . auto . } "

                    forward = np.random.binomial(1, 0.5)
                    if i > 0 and np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.further).capitalize()
                        latex += ", "
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice)
                            latex += np.random.choice(grammar.that)
                        
                        if forward:
                            latex += np.random.choice(grammar.coefficient)
                        else:
                            latex += np.random.choice(grammar.product)
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice).capitalize()
                            latex += np.random.choice(grammar.that)
                            if forward:
                                latex += np.random.choice(grammar.coefficient)
                            else:
                                latex += np.random.choice(grammar.product)
                        elif forward:
                            latex += np.random.choice(grammar.coefficient).capitalize()
                        else:
                            latex += np.random.choice(grammar.product).capitalize()

                    latex += "$ "
                    if forward:
                        latex += nat_to_token(coeffs[order[i]])
                    elif np.random.binomial(1, 0.5):
                        latex += nat_to_token(coeffs[order[i]])
                        latex += times
                        latex += var_to_token(vars[order[i]])
                    else:
                        latex += var_to_token(vars[order[i]])
                        latex += times
                        latex += nat_to_token(coeffs[order[i]])
                    latex += "$ "

                    latex += np.random.choice(grammar.is_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "

                    if forward:
                        latex += ". "
                        latex += np.random.choice(grammar.therefore).capitalize()
                        latex += np.random.choice(grammar.product)
                        latex += "$ "
                        if np.random.binomial(1, 0.5):
                            latex += nat_to_token(coeffs[order[i]])
                            latex += times
                            latex += var_to_token(vars[order[i]])
                        else:
                            latex += var_to_token(vars[order[i]])
                            latex += times
                            latex += nat_to_token(coeffs[order[i]])
                        latex += "$ "
                        latex += np.random.choice(grammar.must_be)
                        latex += "even "
                        latex += np.random.choice(grammar.also)
                        latex += ". "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.since)
                        else:
                            latex += ". "
                            latex += np.random.choice(grammar.this).capitalize()
                            latex += np.random.choice(grammar.holds)
                            latex += np.random.choice(grammar.since)

                        if np.random.binomial(1, 0.5): # State fact about multiplication.
                            latex += np.random.choice(grammar.multiplication)
                            latex += np.random.choice(grammar.of)
                            latex += np.random.choice(grammar.an)
                            latex += "even "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.with_)
                            latex += np.random.choice(grammar.a)
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.must_be)
                            if np.random.binomial(1, 0.5):
                                latex += np.random.choice(grammar.itself)
                                latex += np.random.choice(grammar.an)
                                latex += "even "
                                latex += np.random.choice(grammar.number)
                            else:
                                latex += "even "

                            switch = np.random.randint(3)
                            if switch == 0:
                                latex += np.random.choice(grammar.and_)
                                latex += np.random.choice(grammar.coefficient)
                                latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                                latex += np.random.choice(grammar.is_)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.itself)
                                    latex += np.random.choice(grammar.an)
                                    latex += "even "
                                    latex += np.random.choice(grammar.number)
                                else:
                                    latex += "even "
                                latex += ". "
                            else:
                                if switch == 1:
                                    latex += ", "
                                    latex += np.random.choice(grammar.which)
                                else:
                                    latex += ". "
                                    latex += np.random.choice(grammar.this).capitalize()
                                latex += np.random.choice(grammar.holds)
                                latex += np.random.choice(grammar.since)
                                latex += np.random.choice(grammar.coefficient)
                                latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                                latex += np.random.choice(grammar.is_)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.itself)
                                    latex += np.random.choice(grammar.an)
                                    latex += "even "
                                    latex += np.random.choice(grammar.number)
                                else:
                                    latex += "even "
                                latex += ". "
                        else:
                            latex += np.random.choice(grammar.coefficient)
                            latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                            latex += np.random.choice(grammar.is_)
                            if np.random.binomial(1, 0.5):
                                latex += np.random.choice(grammar.itself)
                                latex += np.random.choice(grammar.an)
                                latex += "even "
                                latex += np.random.choice(grammar.number)
                            else:
                                latex += "even "
                            latex += ". "
            else: # Random length pairs.
                low = 0
                coeffs_order = np.arange(seed[0] - 1)
                np.random.shuffle(coeffs_order)
                while low < seed[0] - 1:
                    high = np.random.randint(low + 1, seed[0] - 1) if low + 1 < seed[0] - 1 else low + 1

                    for i in range(low, high):
                        coq += "assert ( <genH> : Nat . even "
                        coq += nat_to_token(coeffs[coeffs_order[i]])
                        coq += "= true ) . { auto . } "
                    
                    for i in range(low, high):
                        coq += "assert ( <genH> : Nat . even ( "
                        coq += nat_to_token(coeffs[coeffs_order[i]])
                        coq += "* "
                        coq += var_to_token(vars[coeffs_order[i]])
                        coq += ") = true ) . { rewrite Nat . even_mul . rewrite <genH> . auto . } "

                    forward = np.random.binomial(1, 0.5)
                    if low > 0 and np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.further).capitalize()
                        latex += ", "
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice)
                            latex += np.random.choice(grammar.that)
                        
                        if forward and high - low > 1:
                            latex += np.random.choice(grammar.coefficients)
                        elif forward:
                            latex += np.random.choice(grammar.coefficient)
                        elif high - low > 1:
                            latex += np.random.choice(grammar.products)
                        else:
                            latex += np.random.choice(grammar.product)
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.notice).capitalize()
                            latex += np.random.choice(grammar.that)
                            if forward and high - low > 1:
                                latex += np.random.choice(grammar.coefficients)
                            elif forward:
                                latex += np.random.choice(grammar.coefficient)
                            elif high - low > 1:
                                latex += np.random.choice(grammar.products)
                            else:
                                latex += np.random.choice(grammar.product)
                        elif forward and high - low > 1:
                            latex += np.random.choice(grammar.coefficients).capitalize()
                        elif forward:
                            latex += np.random.choice(grammar.coefficient).capitalize()
                        elif high - low > 1:
                            latex += np.random.choice(grammar.products).capitalize()
                        else:
                            latex += np.random.choice(grammar.product).capitalize()

                    for i in range(low, high):
                        if high - low > 1 and i == high - 1:
                            latex += np.random.choice(grammar.and_)
                        latex += "$ "
                        if forward:
                            latex += nat_to_token(coeffs[coeffs_order[i]])
                        elif np.random.binomial(1, 0.5):
                            latex += nat_to_token(coeffs[coeffs_order[i]])
                            latex += times
                            latex += var_to_token(vars[coeffs_order[i]])
                        else:
                            latex += var_to_token(vars[coeffs_order[i]])
                            latex += times
                            latex += nat_to_token(coeffs[coeffs_order[i]])
                        latex += "$ "
                        if high - low > 2 and i < high - 1:
                            latex += ", "

                    if high - low > 1:
                        latex += np.random.choice(grammar.are)
                        if np.random.binomial(1, 0.5):
                            latex += "even "
                            latex += np.random.choice(grammar.numbers)
                        else:
                            latex += "even "
                    else:
                        latex += np.random.choice(grammar.is_)
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.an)
                            latex += "even "
                            latex += np.random.choice(grammar.number)
                        else:
                            latex += "even "

                    if forward:
                        latex += ". "
                        latex += np.random.choice(grammar.therefore).capitalize()
                        if high - low > 1:
                            latex += np.random.choice(grammar.products)
                        else:
                            latex += np.random.choice(grammar.product)
                        for i in range(low, high):
                            if high - low > 1 and i == high - 1:
                                latex += np.random.choice(grammar.and_)
                            latex += "$ "
                            if np.random.binomial(1, 0.5):
                                latex += nat_to_token(coeffs[coeffs_order[i]])
                                latex += times
                                latex += var_to_token(vars[coeffs_order[i]])
                            else:
                                latex += var_to_token(vars[coeffs_order[i]])
                                latex += times
                                latex += nat_to_token(coeffs[coeffs_order[i]])
                            latex += "$ "
                            if high - low > 2 and i < high - 1:
                                latex += ", "
                        latex += np.random.choice(grammar.must_be)
                        latex += "even "
                        latex += np.random.choice(grammar.also)
                        latex += ". "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.since)
                        else:
                            latex += ". "
                            latex += np.random.choice(grammar.this).capitalize()
                            latex += np.random.choice(grammar.holds)
                            latex += np.random.choice(grammar.since)

                        if np.random.binomial(1, 0.5): # State fact about multiplication.
                            latex += np.random.choice(grammar.multiplication)
                            latex += np.random.choice(grammar.of)
                            latex += np.random.choice(grammar.an)
                            latex += "even "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.with_)
                            latex += np.random.choice(grammar.a)
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.must_be)
                            if np.random.binomial(1, 0.5):
                                latex += np.random.choice(grammar.itself)
                                latex += np.random.choice(grammar.an)
                                latex += "even "
                                latex += np.random.choice(grammar.number)
                            else:
                                latex += "even "

                            switch = np.random.randint(3)
                            if switch == 0:
                                latex += np.random.choice(grammar.and_)
                                if high - low > 1:
                                    latex += np.random.choice(grammar.coefficients)
                                else:
                                    latex += np.random.choice(grammar.coefficient)

                                for i in range(low, high):
                                    if high - low > 1 and i == high - 1:
                                        latex += np.random.choice(grammar.and_)
                                    latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                                    if high - low > 2 and i < high - 1:
                                        latex += ", "

                                if high - low > 1:
                                    latex += np.random.choice(grammar.are)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.themselves)
                                        latex += "even "
                                        latex += np.random.choice(grammar.numbers)
                                    else:
                                        latex += "even "
                                else:
                                    latex += np.random.choice(grammar.is_)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.itself)
                                        latex += np.random.choice(grammar.an)
                                        latex += "even "
                                        latex += np.random.choice(grammar.number)
                                    else:
                                        latex += "even "
                                latex += ". "
                            else:
                                if switch == 1:
                                    latex += ", "
                                    latex += np.random.choice(grammar.which)
                                else:
                                    latex += ". "
                                    latex += np.random.choice(grammar.this).capitalize()
                                latex += np.random.choice(grammar.holds)
                                latex += np.random.choice(grammar.since)
                                if high - low > 1:
                                    latex += np.random.choice(grammar.coefficients)
                                else:
                                    latex += np.random.choice(grammar.coefficient)
                                
                                for i in range(low, high):
                                    if high - low > 1 and i == high - 1:
                                        latex += np.random.choice(grammar.and_)
                                    latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                                    if high - low > 2 and i < high - 1:
                                        latex += ", "

                                if high - low > 1:
                                    latex += np.random.choice(grammar.are)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.themselves)
                                        latex += "even "
                                        latex += np.random.choice(grammar.numbers)
                                    else:
                                        latex += "even "
                                else:
                                    latex += np.random.choice(grammar.is_)
                                    if np.random.binomial(1, 0.5):
                                        latex += np.random.choice(grammar.itself)
                                        latex += np.random.choice(grammar.an)
                                        latex += "even "
                                        latex += np.random.choice(grammar.number)
                                    else:
                                        latex += "even "
                                latex += ". "
                        else:
                            if high - low > 1:
                                latex += np.random.choice(grammar.coefficients)
                            else:
                                latex += np.random.choice(grammar.coefficient)
                            
                            for i in range(low, high):
                                if high - low > 1 and i == high - 1:
                                    latex += np.random.choice(grammar.and_)
                                latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                                if high - low > 2 and i < high - 1:
                                    latex += ", "

                            if high - low > 1:
                                latex += np.random.choice(grammar.are)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.themselves)
                                    latex += "even "
                                    latex += np.random.choice(grammar.numbers)
                                else:
                                    latex += "even "
                            else:
                                latex += np.random.choice(grammar.is_)
                                if np.random.binomial(1, 0.5):
                                    latex += np.random.choice(grammar.itself)
                                    latex += np.random.choice(grammar.an)
                                    latex += "even "
                                    latex += np.random.choice(grammar.number)
                                else:
                                    latex += "even "
                            latex += ". "
                    low = high
        elif seed[4] and seed[5]:
            times = np.random.choice(grammar.times)

            for i in range(seed[0]):
                if order[i] == seed[0] - 1:
                    continue
                coq += "assert ( <genH> : Nat . even "
                coq += nat_to_token(coeffs[order[i]])
                coq += "= true ) . { auto . } "

            for i in range(seed[0]):
                if order[i] == seed[0] - 1:
                    continue
                coq += "assert ( <genH> : Nat . even ( "
                coq += nat_to_token(coeffs[order[i]])
                coq += "* "
                coq += var_to_token(vars[order[i]])
                coq += ") = true ) . { rewrite Nat . even_mul . rewrite <genH> . auto . } "

            forward = np.random.binomial(1, 0.5)
            if i > 0 and np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.further).capitalize()
                latex += ", "
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.notice)
                    latex += np.random.choice(grammar.that)
                
                if forward:
                    latex += np.random.choice(grammar.coefficients)
                else:
                    latex += np.random.choice(grammar.products)
            else:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.notice).capitalize()
                    latex += np.random.choice(grammar.that)
                    if forward:
                        latex += np.random.choice(grammar.coefficients)
                    else:
                        latex += np.random.choice(grammar.products)
                elif forward:
                    latex += np.random.choice(grammar.coefficients).capitalize()
                else:
                    latex += np.random.choice(grammar.products).capitalize()

            for i in range(seed[0]):
                if order[i] == seed[0] - 1:
                    continue

                if seed[0] > 2 and i == seed[0] - 1:
                    latex += np.random.choice(grammar.and_)
                latex += "$ "
                if forward:
                    latex += nat_to_token(coeffs[order[i]])
                elif np.random.binomial(1, 0.5):
                    latex += nat_to_token(coeffs[order[i]])
                    latex += times
                    latex += var_to_token(vars[order[i]])
                else:
                    latex += var_to_token(vars[order[i]])
                    latex += times
                    latex += nat_to_token(coeffs[order[i]])
                latex += "$ "
                if seed[0] > 3 and i < seed[0] - 1:
                    latex += ", "

            latex += np.random.choice(grammar.are)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += "even "

            if forward:
                latex += ". "
                latex += np.random.choice(grammar.therefore).capitalize()
                latex += np.random.choice(grammar.products)

                for i in range(seed[0]):
                    if order[i] == seed[0] - 1:
                        continue

                    if seed[0] > 2 and i == seed[0] - 1:
                        latex += np.random.choice(grammar.and_)
                    latex += "$ "
                    if np.random.binomial(1, 0.5):
                        latex += nat_to_token(coeffs[order[i]])
                        latex += times
                        latex += var_to_token(vars[order[i]])
                    else:
                        latex += var_to_token(vars[order[i]])
                        latex += times
                        latex += nat_to_token(coeffs[order[i]])
                    latex += "$ "
                    if seed[0] > 3 and i < seed[0] - 1:
                        latex += ", "

                latex += np.random.choice(grammar.must_be)
                latex += "even "
                latex += np.random.choice(grammar.also)
                latex += ". "
            else:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.since)
                else:
                    latex += ". "
                    latex += np.random.choice(grammar.this).capitalize()
                    latex += np.random.choice(grammar.holds)
                    latex += np.random.choice(grammar.since)

                if np.random.binomial(1, 0.5): # State fact about multiplication.
                    latex += np.random.choice(grammar.multiplication)
                    latex += np.random.choice(grammar.of)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.a)
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.must_be)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.itself)
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "

                    switch = np.random.randint(3)
                    if switch == 0:
                        latex += np.random.choice(grammar.and_)
                        latex += np.random.choice(grammar.coefficients)

                        for i in range(seed[0]):
                            if order[i] == seed[0] - 1:
                                continue

                            if seed[0] > 2 and i == seed[0] - 1:
                                latex += np.random.choice(grammar.and_)
                            latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                            if seed[0] > 3 and i < seed[0] - 1:
                                latex += ", "

                        latex += np.random.choice(grammar.are)
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.themselves)
                            latex += "even "
                            latex += np.random.choice(grammar.numbers)
                        else:
                            latex += "even "
                        latex += ". "
                    else:
                        if switch == 1:
                            latex += ", "
                            latex += np.random.choice(grammar.which)
                        else:
                            latex += ". "
                            latex += np.random.choice(grammar.this).capitalize()
                        latex += np.random.choice(grammar.holds)
                        latex += np.random.choice(grammar.since)
                        latex += np.random.choice(grammar.coefficients)

                        for i in range(seed[0]):
                            if order[i] == seed[0] - 1:
                                continue

                            if seed[0] > 2 and i == seed[0] - 1:
                                latex += np.random.choice(grammar.and_)
                            latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                            if seed[0] > 3 and i < seed[0] - 1:
                                latex += ", "

                        latex += np.random.choice(grammar.are)
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.themselves)
                            latex += "even "
                            latex += np.random.choice(grammar.numbers)
                        else:
                            latex += "even "
                        latex += ". "
                else:
                    latex += np.random.choice(grammar.coefficients)

                    for i in range(seed[0]):
                        if order[i] == seed[0] - 1:
                            continue
                        
                        if seed[0] > 2 and i == seed[0] - 1:
                            latex += np.random.choice(grammar.and_)
                        latex += "$ " + nat_to_token(coeffs[order[i]]) + "$ "
                        if seed[0] > 3 and i < seed[0] - 1:
                            latex += ", "

                    latex += np.random.choice(grammar.are)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.themselves)
                        latex += "even "
                        latex += np.random.choice(grammar.numbers)
                    else:
                        latex += "even "
                    latex += ". "
        elif seed[4]:
            low = 0
            coeffs_order = np.arange(seed[0] - 1)
            np.random.shuffle(coeffs_order)
            while low < seed[0] - 1:
                high = np.random.randint(low + 1, seed[0] - 1) if low + 1 < seed[0] - 1 else low + 1

                for i in range(low, high):
                    coq += "assert ( <genH> : Nat . even "
                    coq += nat_to_token(coeffs[coeffs_order[i]])
                    coq += "= true ) . { auto . } "

                if low > 0 and np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += ", "
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice)
                        latex += np.random.choice(grammar.that)
                    
                    if high - low > 1:
                        latex += np.random.choice(grammar.coefficients)
                    else:
                        latex += np.random.choice(grammar.coefficient)
                else:
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice).capitalize()
                        latex += np.random.choice(grammar.that)
                        if high - low > 1:
                            latex += np.random.choice(grammar.coefficients)
                        else:
                            latex += np.random.choice(grammar.coefficient)
                    elif high - low > 1:
                        latex += np.random.choice(grammar.coefficients).capitalize()
                    else:
                        latex += np.random.choice(grammar.coefficient).capitalize()

                for i in range(low, high):
                    if high - low > 1 and i == high - 1:
                        latex += np.random.choice(grammar.and_)
                    latex += "$ " + nat_to_token(coeffs[coeffs_order[i]]) + "$ "
                    if high - low > 2 and i < high - 1:
                        latex += ", "

                if high - low > 1:
                    latex += np.random.choice(grammar.are)
                    if np.random.binomial(1, 0.5):
                        latex += "even "
                        latex += np.random.choice(grammar.numbers)
                    else:
                        latex += "even "
                else:
                    latex += np.random.choice(grammar.is_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "
                low = high
        elif seed[5]:
            times = np.random.choice(grammar.times)
            
            low = 0
            coeffs_order = np.arange(seed[0] - 1)
            np.random.shuffle(coeffs_order)
            while low < seed[0] - 1:
                high = np.random.randint(low + 1, seed[0] - 1) if low + 1 < seed[0] - 1 else low + 1
                
                for i in range(low, high):
                    coq += "assert ( <genH> : Nat . even ( "
                    coq += nat_to_token(coeffs[coeffs_order[i]])
                    coq += "* "
                    coq += var_to_token(vars[coeffs_order[i]])
                    coq += ") = true ) . { rewrite Nat . even_mul . auto . } "

                if low > 0 and np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += ", "
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice)
                        latex += np.random.choice(grammar.that)
                    
                    if high - low > 1:
                        latex += np.random.choice(grammar.products)
                    else:
                        latex += np.random.choice(grammar.product)
                else:
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.notice).capitalize()
                        latex += np.random.choice(grammar.that)
                        if high - low > 1:
                            latex += np.random.choice(grammar.products)
                        else:
                            latex += np.random.choice(grammar.product)
                    elif high - low > 1:
                        latex += np.random.choice(grammar.products).capitalize()
                    else:
                        latex += np.random.choice(grammar.product).capitalize()

                for i in range(low, high):
                    if high - low > 1 and i == high - 1:
                        latex += np.random.choice(grammar.and_)
                    latex += "$ "
                    if np.random.binomial(1, 0.5):
                        latex += nat_to_token(coeffs[coeffs_order[i]])
                        latex += times
                        latex += var_to_token(vars[coeffs_order[i]])
                    else:
                        latex += var_to_token(vars[coeffs_order[i]])
                        latex += times
                        latex += nat_to_token(coeffs[coeffs_order[i]])
                    latex += "$ "
                    if high - low > 2 and i < high - 1:
                        latex += ", "

                if high - low > 1:
                    latex += np.random.choice(grammar.are)
                    if np.random.binomial(1, 0.5):
                        latex += "even "
                        latex += np.random.choice(grammar.numbers)
                    else:
                        latex += "even "
                else:
                    latex += np.random.choice(grammar.is_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "

                if np.random.binomial(1, 0.5):
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.since)
                    else:
                        latex += ". "
                        latex += np.random.choice(grammar.this).capitalize()
                        latex += np.random.choice(grammar.holds)
                        latex += np.random.choice(grammar.since)

                    latex += np.random.choice(grammar.multiplication)
                    latex += np.random.choice(grammar.of)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.a)
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.must_be)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.itself)
                        latex += np.random.choice(grammar.an)
                        latex += "even "
                        latex += np.random.choice(grammar.number)
                    else:
                        latex += "even "
                latex += ". "
                low = high
        elif not seed[9] and not seed[14] and not seed[16]:
            latex += np.random.choice(grammar.theorem).capitalize()
            latex += np.random.choice(grammar.trivially)
            latex += np.random.choice(grammar.holds)
            latex += ". "

    if not (restatement_start or addition_start or multiplication_start) and (restatement_end or addition_end or multiplication_end) and seed[3] == 1:
        const_rewrite = True
        coq += "assert ( <genH> : Nat . "
        coq += ["even ", "odd "][seed[1]]
        coq += nat_to_token(const)
        coq += "= true ) . { auto . } "

        latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += "$ " + nat_to_token(const) + "$ "
        latex += np.random.choice(grammar.is_)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]
        latex += ". "

    # Restatement, addition phrase, and multiplication phrase all at start.
    if restatement_end and addition_end and multiplication_end:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        switch = np.random.randint(3)
        if switch == 0: # Rules imply th'm.
            latex += np.random.choice(grammar.since).capitalize()
            latex += np.random.choice(grammar.addition)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]

            latex += np.random.choice(grammar.and_)
            latex += np.random.choice(["", np.random.choice(grammar.since)])
            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.a)
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ", "

            latex += np.random.choice(grammar.then)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
                latex += ". "
        elif switch == 1: # Rules, therefore th'm.
            latex += np.random.choice(grammar.addition).capitalize()
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]
            
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.and_)
            else:
                latex += ". "
                latex += np.random.choice(grammar.further).capitalize()

            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ". "

            latex += np.random.choice(grammar.therefore).capitalize()
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # Th'm, then rules.
            if np.random.binomial(1, 0.5): # We show th'm by rules.
                latex += np.random.choice(grammar.show).capitalize()
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression)
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, const, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += ["even ", "odd "][seed[1]]
                else:
                    latex += np.random.choice(grammar.our)
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
                latex += np.random.choice(grammar.using)
                latex += np.random.choice(grammar.fact)
                latex += np.random.choice(grammar.that)

                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]

                latex += np.random.choice(grammar.and_)
                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
            else: # Th'm. Rules.
                latex += np.random.choice(grammar.show).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "

                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.first).capitalize()
                    latex += np.random.choice(grammar.notice)
                else:
                    latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]

                if np.random.binomial(1, 0.5):
                    latex += ", "
                    latex += np.random.choice(grammar.and_)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.next)
                    latex += np.random.choice(grammar.notice)
                else:
                    latex += ". "
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.next).capitalize()
                        latex += np.random.choice(grammar.notice)
                    else:
                        latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
    # Restatement and addition phrase at start.
    elif restatement_end and addition_end:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            if not sum_of_prods:
                coq += "rewrite <- Nat . negb_even . "
                coq += "repeat rewrite Nat . even_add . "
        else:
            if not sum_of_prods:
                coq += "repeat "
            coq += "rewrite Nat . even_add . "
        
        switch = np.random.randint(3)
        if switch == 0: # Rules imply th'm.
            latex += np.random.choice(grammar.since).capitalize()
            latex += np.random.choice(grammar.addition)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]
            latex += ", "

            latex += np.random.choice(grammar.then)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
                latex += ". "
        elif switch == 1: # Rules, therefore th'm.
            latex += np.random.choice(grammar.addition).capitalize()
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
                latex += np.random.choice(grammar.together)
                if seed[1] or np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
            else:
                latex += ["even ", "odd "][seed[1]]
            latex += ". "

            latex += np.random.choice(grammar.therefore).capitalize()
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # Th'm, then rules.
            if np.random.binomial(1, 0.5): # We show th'm by rules.
                latex += np.random.choice(grammar.show).capitalize()
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression)
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, const, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += ["even ", "odd "][seed[1]]
                else:
                    latex += np.random.choice(grammar.our)
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
                latex += np.random.choice(grammar.using)
                latex += np.random.choice(grammar.fact)
                latex += np.random.choice(grammar.that)

                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else: # Th'm. Rules.
                latex += np.random.choice(grammar.show).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "

                latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.addition)
                latex += np.random.choice(grammar.of)
                if np.random.binomial(1, 0.5):
                    latex += "even "
                    latex += np.random.choice(grammar.numbers)
                    latex += np.random.choice(grammar.together)
                    if seed[1] or np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.with_)
                        latex += np.random.choice(grammar.an)
                        latex += ["even ", "odd "][seed[1]]
                        latex += np.random.choice(grammar.number)
                else:
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.with_)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += ["even ", "odd "][seed[1]]
                    latex += np.random.choice(grammar.number)
                else:
                    latex += ["even ", "odd "][seed[1]]
                latex += ". "
    # Restatement and multiplication phrase at start.
    elif restatement_end and multiplication_end:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        switch = np.random.randint(3)
        if switch == 0: # Rules imply th'm.
            latex += np.random.choice(grammar.since).capitalize()
            latex += np.random.choice(grammar.multiplication)
            latex += np.random.choice(grammar.of)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.a)
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ", "

            latex += np.random.choice(grammar.then)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
                latex += ". "
        elif switch == 1: # Rules, therefore th'm.
            latex += np.random.choice(grammar.multiplication).capitalize()
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5):
                latex += "even "
                latex += np.random.choice(grammar.numbers)
            else:
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.must_be)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
            else:
                latex += "even "
            latex += ". "

            latex += np.random.choice(grammar.therefore).capitalize()
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.itself)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # Th'm, then rules.
            if np.random.binomial(1, 0.5): # We show th'm by rules.
                latex += np.random.choice(grammar.show).capitalize()
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.expression)
                    np.random.shuffle(order)
                    latex += expression(coeffs, vars, const, order, latex=True)
                    latex += np.random.choice(grammar.must_be)
                    latex += ["even ", "odd "][seed[1]]
                else:
                    latex += np.random.choice(grammar.our)
                    latex += np.random.choice(grammar.theorem)
                    latex += np.random.choice(grammar.holds)
                latex += np.random.choice(grammar.using)
                latex += np.random.choice(grammar.fact)
                latex += np.random.choice(grammar.that)

                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
            else: # Th'm. Rules.
                latex += np.random.choice(grammar.show).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.expression)
                np.random.shuffle(order)
                latex += expression(coeffs, vars, const, order, latex=True)
                latex += np.random.choice(grammar.must_be)
                latex += ["even ", "odd "][seed[1]]
                latex += ". "

                latex += np.random.choice(grammar.notice).capitalize()
                latex += np.random.choice(grammar.that)
                latex += np.random.choice(grammar.multiplication)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.an)
                latex += "even "
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.a)
                latex += np.random.choice(grammar.number)
                latex += np.random.choice(grammar.must_be)
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.itself)
                    latex += np.random.choice(grammar.an)
                    latex += "even "
                    latex += np.random.choice(grammar.number)
                else:
                    latex += "even "
                latex += ". "
    # Addition and multiplication phrases at start.
    elif addition_end and multiplication_end:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.first).capitalize()
            latex += np.random.choice(grammar.notice)
        else:
            latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += np.random.choice(grammar.addition)
        latex += np.random.choice(grammar.of)
        if np.random.binomial(1, 0.5):
            latex += "even "
            latex += np.random.choice(grammar.numbers)
            latex += np.random.choice(grammar.together)
            if seed[1] or np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
        else:
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]

        if np.random.binomial(1, 0.5):
            latex += ", "
            latex += np.random.choice(grammar.and_)
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.next)
            latex += np.random.choice(grammar.notice)
        else:
            latex += ". "
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.next).capitalize()
                latex += np.random.choice(grammar.notice)
            else:
                latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += np.random.choice(grammar.multiplication)
        latex += np.random.choice(grammar.of)
        latex += np.random.choice(grammar.an)
        latex += "even "
        latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.with_)
        latex += np.random.choice(grammar.a)
        latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
        else:
            latex += "even "
        latex += ". "
    # Restatement at start.
    elif restatement_end:
        latex += np.random.choice(grammar.show).capitalize()
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.expression)
            np.random.shuffle(order)
            latex += expression(coeffs, vars, const, order, latex=True)
            latex += np.random.choice(grammar.must_be)
            latex += ["even ", "odd "][seed[1]]
        else:
            latex += np.random.choice(grammar.our)
            latex += np.random.choice(grammar.theorem)
            latex += np.random.choice(grammar.holds)
        latex += ". "
    # Addition phrase at start.
    elif addition_end:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            if not sum_of_prods:
                coq += "rewrite <- Nat . negb_even . "
                coq += "repeat rewrite Nat . even_add . "
        else:
            if not sum_of_prods:
                coq += "repeat "
            coq += "rewrite Nat . even_add . "

        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.notice).capitalize()
            latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.addition)
        else:
            latex += np.random.choice(grammar.addition).capitalize()
        latex += np.random.choice(grammar.of)
        if np.random.binomial(1, 0.5):
            latex += "even "
            latex += np.random.choice(grammar.numbers)
            latex += np.random.choice(grammar.together)
            if seed[1] or np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.with_)
                latex += np.random.choice(grammar.an)
                latex += ["even ", "odd "][seed[1]]
                latex += np.random.choice(grammar.number)
        else:
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]
        latex += ". "
    # Multiplication phrase at start.
    elif multiplication_end:
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "

        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.notice).capitalize()
            latex += np.random.choice(grammar.that)
            latex += np.random.choice(grammar.multiplication)
        else:
            latex += np.random.choice(grammar.multiplication).capitalize()
        latex += np.random.choice(grammar.of)
        if np.random.binomial(1, 0.5):
            latex += "even "
            latex += np.random.choice(grammar.numbers)
        else:
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
            latex += np.random.choice(grammar.with_)
            latex += np.random.choice(grammar.a)
            latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.must_be)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.itself)
            latex += np.random.choice(grammar.an)
            latex += "even "
            latex += np.random.choice(grammar.number)
        else:
            latex += "even "
        latex += ". "

    if not (restatement_start or addition_start or multiplication_start) and (restatement_end or addition_end or multiplication_end) and seed[3] == 2:
        const_rewrite = True
        coq += "assert ( <genH> : Nat . "
        coq += ["even ", "odd "][seed[1]]
        coq += nat_to_token(const)
        coq += "= true ) . { auto . } "

        latex += np.random.choice(grammar.notice).capitalize()
        latex += np.random.choice(grammar.that)
        latex += "$ " + nat_to_token(const) + "$ "
        latex += np.random.choice(grammar.is_)
        if np.random.binomial(1, 0.5):
            latex += np.random.choice(grammar.an)
            latex += ["even ", "odd "][seed[1]]
            latex += np.random.choice(grammar.number)
        else:
            latex += ["even ", "odd "][seed[1]]
        latex += ". "

    if sum_of_prods:
        if not addition_start and not addition_end:
            coq += "rewrite Nat . "
            coq += ["even_add ", "odd_add "][seed[1]]
            coq += ". "
        if seed[1]:
            coq += "rewrite <- Nat . negb_even . "
        coq += "rewrite <genH> . "
    elif seed[5]:
        if not addition_start and not addition_end:
            if seed[1]:
                coq += "rewrite Nat . odd_add . "
                coq += "rewrite <- Nat . negb_even . "
            coq += "repeat rewrite Nat . even_add . "

        for i in range(seed[0] - 1):
            coq += "rewrite <genH> . "
    elif seed[4]:
        if not addition_start and not addition_end:
            if seed[1]:
                coq += "rewrite Nat . odd_add . "
                coq += "rewrite <- Nat . negb_even . "
            coq += "repeat rewrite Nat . even_add . "

        if not multiplication_start and not multiplication_end:
            coq += "repeat rewrite Nat . even_mul . "

        for i in range(seed[0] - 1):
            coq += "rewrite <genH> . "
    elif not (multiplication_start or multiplication_end) and not (addition_start or addition_end):
        if seed[1]:
            coq += "rewrite Nat . odd_add . "
            coq += "rewrite <- Nat . negb_even . "
        coq += "repeat rewrite Nat . even_add . "
        coq += "repeat rewrite Nat . even_mul . "
    elif (addition_start or addition_end) and not (multiplication_start or multiplication_end):
        coq += "repeat rewrite Nat . even_mul . "
    if const_rewrite:
        coq += "rewrite <genH> . "
    coq += "auto . "

    latex += r"\end{proof} \end{document} "
    coq += "Qed . "

    # Write to file.

    f_sum = "_"
    for idx in order:
        if idx != len(coeffs):
            f_sum += str(coeffs[idx]) + vars[idx] + "_"
    f_sum += str(const)
    filename = args.path + "/" + args.set + "/even-odd" + f_sum + "_tokenized.txt"

    if args.debug:
        detok_coq = detokenize_coq(coq)
        proof, _ = check_proof(detok_coq, "")

        print("Example " + str(count) + ": ", end="")
        print(seed)
        print(latex)
        print(detok_coq)
        print("")
        if True: #not proof:
            input("Invalid proof.")
    else:
        doc = open(filename, "w")
        doc.write(latex + "\n~\n" + coq)
        doc.close()

def powers(count):
    grammar = Powers()

    num_options = {
        'training': [2, 3, 5, 7, 9],
        'validation': [2, 3, 4, 5, 7, 8, 9],
        'test': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }

    seed = [
        np.random.choice(num_options[args.set]),    #  0: Number of factors.
        np.random.binomial(1, 0.5),                 #  1: Definition.
        np.random.binomial(1, 0.5),                 #  2: Define term at beginning of definition.
        np.random.binomial(1, 0.5),                 #  3: Set notation in theorem.
        np.random.binomial(1, 0.5),                 #  4: Logic notation (\exists).
        np.random.binomial(1, 0.5),                 #  5: Variable in theorem.
        np.random.randint(3),                       #  6: Lower bound condition. (0) >= 2, (1) > 1, (2) none.
        np.random.binomial(1, 0.5),                 #  7: If [6] < 2, bounds before power.
        np.random.binomial(1, 0.5),                 #  8: If not [2], definition before name.
        np.random.binomial(1, 0.5),                 #  9: Let statement in proof.
        np.random.binomial(1, 0.5),                 # 10: Restatement of theorem.
        np.random.binomial(1, 0.5),                 # 11: If [1], restatement of definition.
        np.random.binomial(1, 0.5),                 # 12: If [10], state at (1) start or (0) end of proof.
        np.random.binomial(1, 0.5),                 # 13: If [6] and [10], include restatement of lower bound.
        np.random.binomial(1, 0.5),                 # 14: If [6], prove lower bound.
        np.random.binomial(1, 0.5),                 # 15: If [6] and [14], prove lower bound (1) before or (0) after factor proof.
        np.random.binomial(1, 0.5),                 # 16: Directional reasoning.
        np.random.binomial(1, 0.5),                 # 17: If [16], then (1) forward (a^2 = x -> th'm) or (0) backward (th'm, b.c. a^2 = x).
        np.random.binomial(1, 0.5),                 # 18: Intermediate algebraic steps. i.e. a^2 = 2^2 = 4.
        np.random.binomial(1, 0.5),                 # 19: If [9], don't use constants. i.e. 4 = a^2 not 4 = 2^2.
        np.random.binomial(1, 0.5),                 # 20: If not [9], factors imply variable values. i.e. 4 = 2^2 -> a = 2.
        np.random.binomial(1, 0.5),                 # 21: If [9], no equations.
        np.random.binomial(1, 0.5),                 # 22: If [1], include definition in theorem.
        np.random.binomial(1, 0.5),                 # 23: If [1], have "x = 4" in theorem.
    ]

    base = 2
    power = base ** seed[0]
    # base = np.random.randint(2, 11)
    # power = base ** seed[0]
    # while power >= 32768:
    #     base = np.random.randint(10)
    #     seed[0] = np.random.choice(num_options[args.set])
    vars = np.array(list(string.ascii_letters))
    np.random.shuffle(vars)
    vars = vars[:2]

    defs = [
        "",
        "",
        "square",
        "cube",
        "tesseract",
        "penteract",
        "hexeract",
        "hepteract",
        "octeract",
        "enneract",
        "dekeract",
        "hendekeract",
        "dodekeract"
    ]

    latex = r"\documentclass[12pt]{article} \usepackage{amsthm,amsfonts,amssymb} \newtheorem{theorem}{Theorem} "
    if seed[1] and not seed[9]:
        latex += r"\newtheorem{definition}{Definition} "
    latex += r"\begin{document} "

    coq = "Require Import Arith . Require Import Lia . "

    if seed[1]:
        coq += "Definition <def:" + defs[seed[0]] + "> ( " + var_to_token(vars[-1]) + ": nat ) := "
        coq += "exists "
        coq += var_to_token(vars[0])
        coq += ": nat , "

    # Definition
    if seed[1] and not seed[9]: # Explicit definition.
        latex += r"\begin{definition} "
        if seed[2]: # Let a at start of definition.
            if seed[3] and seed[4]: # Exists a in Z.
                if seed[6] < 2 and seed[7]:
                    latex += np.random.choice(grammar.exists).capitalize()
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                    latex += np.random.choice(grammar.exists)
                latex += "$ " + var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[-1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[3]: # Let a in Z.
                latex += np.random.choice(grammar.let).capitalize()
                latex += "$ " + var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[4]: # Exists a.
                if seed[6] < 2 and seed[7]:
                    latex += np.random.choice(grammar.exists[:-1])[:-2].capitalize()
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                    latex += np.random.choice(grammar.exists[:-1])[:-2]
                latex += np.random.choice(grammar.number)
                latex += "$ " + var_to_token(vars[0]) + "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
            else: # Let a.
                term = np.random.choice(grammar.let)
                latex += term.capitalize()
                latex += "$ " + var_to_token(vars[0]) + "$ "
                if term == "let ":
                    latex += grammar.be[0]
                else:
                    latex += np.random.choice(grammar.be[1:])
                latex += np.random.choice(grammar.numbers[:-1])
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "base "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[-1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
        else: # Let a later in definition.
            if seed[8]: # Definition before <def> token.
                latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.have)
                if seed[3]: # a in Z.
                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                else: # Let a.
                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
            else: # <def> token, then definition.
                if seed[3] and seed[4]: # Exists a in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exists)
                    latex += var_to_token(vars[0])
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[3]: # Let a in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += "$ " + var_to_token(vars[0])
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[4]: # Exists a.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exists[:-1])[:-2]
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.number)
                    latex += "$ " + var_to_token(vars[0]) + "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                else: # Let a.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.number)
                    latex += "$ " + var_to_token(vars[0]) + "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
        latex += r"\end{definition} "

    # Theorem
    latex += r"\begin{theorem} "
    if seed[1] and seed[9]: # Definition in theorem.
        if seed[2]: # Let a at start of definition.
            if seed[3] and seed[4]: # Exists a in Z.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += np.random.choice(grammar._if)
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.exists)
                latex += var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[3]: # Let a in Z.
                latex += np.random.choice(grammar.let).capitalize()
                latex += "$ " + var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[4]: # Exists a.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += np.random.choice(grammar._if)
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.exists[:-1])[:-2]
                latex += np.random.choice(grammar.number)
                latex += "$ " + var_to_token(vars[0]) + "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
            else: # Let a.
                term = np.random.choice(grammar.let)
                latex += term.capitalize()
                latex += "$ " + var_to_token(vars[0]) + "$ "
                if term == "let ":
                    latex += grammar.be[0]
                else:
                    latex += np.random.choice(grammar.be[1:])
                latex += np.random.choice(grammar.numbers[:-1])
                
                if seed[6] < 2 and seed[7]: # Bounds before power.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do power.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.base)
                        latex += "is "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                        else:
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do power.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar.when)
                    
                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    else:
                        latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                    latex += "$ . "
                    coq += ") . "
        else: # Let a later in definition.
            if seed[8]: # Definition before <def> token.
                latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.have)
                if seed[3]: # a in Z.
                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                else: # Let a.
                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\geq <nat:2> $ "
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0]) + "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[0])
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            latex += "a <def:" + defs[seed[0]] + "> "
                            latex += np.random.choice(grammar.number)
                        latex += ". "
                        coq += ". "
            else: # <def> token, then definition.
                if seed[3] and seed[4]: # Exists a in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exists)
                    latex += var_to_token(vars[0])
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[3]: # Let a in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += "$ " + var_to_token(vars[0])
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[4]: # Exists a.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exists[:-1])[:-2]
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.number)
                    latex += "$ " + var_to_token(vars[0]) + "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            latex += r"\geq <nat:2> $ "
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                else: # Let a.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    latex += "a <def:" + defs[seed[0]] + "> "
                    latex += np.random.choice(grammar.number)
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.number)
                    latex += "$ " + var_to_token(vars[0]) + "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before power.
                        if seed[6] == 0: # >= 2.
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ " + var_to_token(vars[0])
                            coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                            latex += r"> <nat:1> $ "

                        # Do power.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ , "
                        
                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ " + var_to_token(vars[0])
                        coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do power.
                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                            coq += var_to_token(vars[1]) + "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                        else:
                            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                            coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0]) + "= " + var_to_token(vars[1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[1]) + "$ "
                            
                        latex += ". "
                        coq += ". "

        latex += np.random.choice(grammar.so).capitalize()
        if seed[10] and np.random.binomial(1, 0.5):
            latex += "$ "
            latex += var_to_token(vars[1])
            latex += "= "
            latex += nat_to_token(power)
            latex += "$ "
        elif seed[10]:
            latex += "$ "
            latex += nat_to_token(power)
            latex += "= "
            latex += var_to_token(vars[1])
            latex += "$ "
        else:
            term = np.random.choice(grammar.number)
            if term != "":
                latex += "the "
            latex += term
            latex += nat_to_token(power)
        latex += np.random.choice(grammar._is)
        latex += "a <def:" + defs[seed[0]] + "> "
        latex += np.random.choice(grammar.number)
        latex += ". "

        coq += "Theorem <genP:1> : <def:" + defs[seed[0]] + "> " + nat_to_token(power) + ". "
    elif seed[1]: # Explicit definition.
        coq += "Theorem <genP:1> : <def:" + defs[seed[0]] + "> " + nat_to_token(power) + ". "

        if seed[10] and np.random.binomial(1, 0.5):
            latex += "$ "
            latex += var_to_token(vars[1])
            latex += "= "
            latex += nat_to_token(power)
            latex += "$ "
        elif seed[10]:
            latex += "$ "
            latex += nat_to_token(power)
            latex += "= "
            latex += var_to_token(vars[1])
            latex += "$ "
        else:
            term = np.random.choice(grammar.number)
            if term != "":
                latex += "The "
            latex += term
            latex += nat_to_token(power)
        latex += np.random.choice(grammar._is)
        latex += "a <def:" + defs[seed[0]] + "> "
        latex += np.random.choice(grammar.number)
        latex += ". "
    else: # No definition.
        coq += "Theorem <genP:1> : exists "
        coq += var_to_token(vars[0])
        coq += ": nat , "

        if seed[2]: # 4 = a^2 for some ...
            if seed[10] and np.random.binomial(1, 0.5):
                latex += "$ "
                coq += "( "

                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += np.random.choice(grammar.equal)
                latex += nat_to_token(power)
                latex += "$ "

                coq += "= " + nat_to_token(power) + ") "
            elif seed[10]:
                latex += "$ "
                latex += nat_to_token(power)
                latex += np.random.choice(grammar.equal)
                coq += "( " + nat_to_token(power) + "= "

                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "$ "
                coq += ") "
            else:
                term = np.random.choice(grammar.number)
                if term != "":
                    latex += "The "
                latex += term
                latex += nat_to_token(power)
                latex += np.random.choice(grammar._is)
                latex += np.random.choice(grammar.exponent)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.number)

                coq += "( " + nat_to_token(power) + "= "
                latex += "$ " + var_to_token(vars[0])
                coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                if seed[3]:
                    latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                coq += ") "

                latex += np.random.choice(grammar.raised)
                latex += np.random.choice(grammar.power)
                latex += nat_to_token(seed[0])

            if seed[10] and seed[5]: # For some a.
                latex += np.random.choice(grammar._for)
                latex += "$ " + var_to_token(vars[0])
                if seed[3]:
                    latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "

            if seed[6] == 0:
                latex += np.random.choice(grammar.such_that)
                latex += "$ "
                coq += "/\ "
                latex += var_to_token(vars[0])
                coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            elif seed[6] == 1:
                latex += np.random.choice(grammar.such_that)
                latex += "$ "
                coq += "/\ "
                latex += var_to_token(vars[0])
                coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += ". "
            coq += ". "
        else: # For some ... 4 = a^2.
            if seed[3] and seed[4]:
                latex += np.random.choice(grammar.exists).capitalize()
                latex += var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            elif seed[3]:
                latex += np.random.choice(grammar.let).capitalize()
                latex += "$ " + var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            elif seed[4]:
                term = np.random.choice(grammar.exists)
                if "there" in term:
                    latex += term[:-2].capitalize()
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.number)
                    latex += "$ "
                else:
                    latex += term.capitalize()
                latex += var_to_token(vars[0])
                latex += "$ "
            else:
                term = np.random.choice(grammar.let)
                latex += term.capitalize()
                latex += "$ " + var_to_token(vars[0]) + "$ "
                if term == "let ":
                    latex += grammar.be[0]
                else:
                    latex += np.random.choice(grammar.be[1:])
                latex += np.random.choice(grammar.numbers[:-1])

            if seed[6] == 0:
                latex += np.random.choice(grammar.such_that)
                latex += "$ " + var_to_token(vars[0])
                coq += "( " + var_to_token(vars[0]) + ">= <nat:2> ) /\ "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            elif seed[6] == 1:
                latex += np.random.choice(grammar.such_that)
                latex += "$ " + var_to_token(vars[0])
                coq += "( " + var_to_token(vars[0]) + "> <nat:1> ) /\ "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            
            if seed[4]:
                if seed[6] < 2:
                    latex += np.random.choice(grammar._and)
                latex += np.random.choice(grammar.such_that)
            else:
                term = np.random.choice(grammar.such_that + [". "])
                if seed[6] < 2 and term != ". ":
                    latex += np.random.choice(grammar._and)
                    latex += term
                else:
                    latex += term

            if seed[10] and np.random.binomial(1, 0.5):
                latex += "$ "
                coq += "( "

                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += np.random.choice(grammar.equal)
                latex += nat_to_token(power)
                latex += "$ "

                coq += "= " + nat_to_token(power) + ") "
            elif seed[10]:
                latex += "$ "
                latex += nat_to_token(power)
                latex += np.random.choice(grammar.equal)
                coq += "( " + nat_to_token(power) + "= "

                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "$ "
                coq += ") "
            else:
                term = np.random.choice(grammar.number)
                if term != "":
                    latex += "the "
                latex += term
                latex += nat_to_token(power)
                latex += np.random.choice(grammar._is)
                latex += np.random.choice(grammar.exponent)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.number)

                coq += "( " + nat_to_token(power) + "= "
                latex += "$ "
                latex += var_to_token(vars[0])
                coq += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                if seed[3]:
                    latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                coq += ") "

                latex += np.random.choice(grammar.raised)
                latex += np.random.choice(grammar.power)
                latex += nat_to_token(seed[0])
            latex += ". "
            coq += ". "
    latex += r"\end{theorem} "

    # Proof
    latex += r"\begin{proof} "
    coq += "Proof . "

    if seed[1]:
        coq += "unfold <def:" + defs[seed[0]] + "> . "

    if seed[13] and seed[1]: # Restate definition.
        latex += np.random.choice(grammar.recall).capitalize()
        if np.random.binomial(1, 0.5):
            latex += "a <def:" + defs[seed[0]] + "> "
            latex += np.random.choice(grammar.number)
        else:
            latex += nat_to_token(power)
            latex += np.random.choice(grammar._is)
            term = np.random.choice(grammar.number)
            if term != "":
                latex += "a "
            latex += "<def:" + defs[seed[0]] + "> "
            latex += term
            latex += np.random.choice(grammar._if)
            latex += np.random.choice(["it ", nat_to_token(power)])
        latex += np.random.choice(grammar._is)
        latex += np.random.choice(grammar.exponent)
        latex += np.random.choice(grammar.of)
        latex += np.random.choice(grammar.number)
        latex += "$ " + var_to_token(vars[0]) + "$ "
        latex += np.random.choice(grammar.raised)
        latex += np.random.choice(grammar.power)
        latex += nat_to_token(seed[0])

        if seed[6] == 0: # >= 2.
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.greater)
                latex += np.random.choice(grammar.equal_to)
                latex += np.random.choice(grammar.two)
            else:
                latex += np.random.choice(grammar.such_that)
                latex += "$ " + var_to_token(vars[0])
                latex += r"\geq <nat:2> $ "
        else: # > 1.
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.greater)
                latex += np.random.choice(grammar.one)
            else:
                latex += np.random.choice(grammar.such_that)
                latex += "$ " + var_to_token(vars[0])
                latex += r"> <nat:1> $ "
        latex += ". "

    if seed[11]: # Let a = 2.
        latex += np.random.choice(grammar.let).capitalize()
        latex += "$ " + var_to_token(vars[0]) + "= " + nat_to_token(base)
        coq += "exists " + nat_to_token(base) + ". "
        latex += "$ . "

    if seed[12] and seed[14] and seed[13] and seed[1]: # Restate theorem to be proven, just tag.
        latex += np.random.choice(grammar.show).capitalize()
        latex += nat_to_token(power)
        latex += np.random.choice(grammar._is)
        term = np.random.choice(grammar.number)
        if term != "":
            latex += "a "
        latex += "<def:" + defs[seed[0]] + "> "
        latex += term
        latex += ". "
    elif seed[12] and seed[14]: # Restate theorem, no tag.
        latex += np.random.choice(grammar.show).capitalize()
        if seed[15] and seed[6] < 2 and seed[7]: # Include bounds before restatement.
            latex += "$ "
            if seed[6] == 0: # >= 2.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += np.random.choice(grammar._and)
        
        switch = np.random.randint(3)
        if switch == 0: # 4 = a^2.
            latex += "$ "
            latex += nat_to_token(power)
            latex += "= "
            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            latex += "$ "
        elif switch == 1: # a^2 = 4.
            latex += "$ "
            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            latex += "= "
            latex += nat_to_token(power)
            latex += "$ "
        else: # 6 is power.
            latex += nat_to_token(power)
            latex += np.random.choice(grammar._is)
            latex += np.random.choice(grammar.exponent)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5): # a in Z.
                latex += "$ " + var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            else: # Number a.
                latex += np.random.choice(grammar.some)
                latex += np.random.choice(grammar.number)
                latex += "$ " + var_to_token(vars[0]) + "$ "
            latex += np.random.choice(grammar.raised)
            latex += np.random.choice(grammar.power)
            latex += nat_to_token(seed[0])

        if seed[15] and seed[6] < 2 and not seed[7]: # Include bounds after restatement.
            latex += np.random.choice(grammar._and)
            latex += "$ "
            if seed[6] == 0: # >= 2.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
        latex += ". "

    if seed[18]: # Directional reasoning. i.e. 4 = 2^2 -> th'm.
        if seed[19]: # 4 = 2^2 -> th'm.
            if seed[16] and seed[6] < 2 and seed[17]: # Prove bounds before power.
                latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
                latex += ". "
            
            if seed[11] and seed[23]: # No equations.
                latex += ""
            elif seed[11] and seed[21]: # 4 = a^2, no constants.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                if np.random.binomial(1, 0.5): # 4 = a^2.
                    latex += nat_to_token(power)
                    latex += "= " + var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                else: # a^2 = 6.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                latex += "$ . "
            elif not seed[11] and seed[22]: # 4 = 2^2 -> a = 2.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                if np.random.binomial(1, 0.5): # 4 = a^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                else: # a^2 = 6.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                latex += "$ "
                latex += np.random.choice(grammar.gives)
                latex += "$ " + var_to_token(vars[0]) + "= " + nat_to_token(base)
                
                coq += "exists " + nat_to_token(base) + ". "
                latex += "$ . "
            elif seed[20]: # 4 = a^2 = 2^2.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                switch = np.random.randint(6)
                if switch == 0: # 4 = a^2 = 2^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                elif switch == 1: # 4 = 2^2 = a^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                elif switch == 2: # a^2 = 2^2 = 4.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                elif switch == 3: # a^2 = 4 = 2^2.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                elif switch == 4: # 2^2 = a^2 = 4.
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                else: # 2^2 = 4 = a^2.
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "$ . "

                if not seed[11]:
                    coq += "exists " + nat_to_token(base) + ". "
            else: # 4 = 2^2.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                if np.random.binomial(1, 0.5): # 4 = 2^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                else: # 2^2 = 4.
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                latex += "$ . "

                if not seed[11]:
                    coq += "exists " + nat_to_token(base) + ". "
            if seed[16] and seed[6] < 2 and not seed[17]: # Prove bounds after power.
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += "$ "
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
                latex += ". "

            latex += np.random.choice(grammar.this).capitalize()
            latex += np.random.choice(grammar.gives)
            if seed[1] and np.random.binomial(1, 0.5): # 4 is composite.
                latex += nat_to_token(power)
                latex += np.random.choice(grammar._is)
                latex += "a <def:" + defs[seed[0]] + "> "
                latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # th'm b/c 4 = 2^2
            latex += np.random.choice(grammar.prove).capitalize()
            if seed[1] and np.random.binomial(1, 0.5): # 6 is composite.
                latex += nat_to_token(power)
                latex += np.random.choice(grammar._is)
                latex += "a <def:" + defs[seed[0]] + "> "
                latex += np.random.choice(grammar.number)
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += np.random.choice(grammar.since)

            if seed[16] and seed[6] < 2 and seed[17]: # Prove bounds before power.
                latex += "$ "
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
            
            if seed[11] and seed[23] and seed[16] and seed[6] < 2: # No equations.
                latex += ""
            elif seed[11] and seed[21]: # 4 = a^2, no constants.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                if np.random.binomial(1, 0.5): # 4 = a^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                else: # a^2 = 6.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                latex += "$ "
            elif not seed[11] and seed[22]: # 4 = 2^2 -> a = 2.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                if np.random.binomial(1, 0.5): # 4 = a^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                else: # a^2 = 4.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                latex += "$ "
                latex += np.random.choice(grammar.gives)
                latex += "$ "
                latex += var_to_token(vars[0])
                latex += "= "
                latex += nat_to_token(base)

                coq += "exists " + nat_to_token(base) + ". "
                latex += "$ "
            elif seed[20]: # 4 = a^2 = 2^2.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                switch = np.random.randint(6)
                if switch == 0: # 6 = a^2 = 2^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                elif switch == 1: # 4 = 2^2 = a^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                elif switch == 2: # a^2 = 2^2 = 4.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                elif switch == 3: # a^2 = 4 = 2^2.
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                elif switch == 4: # 2^2 = a^2 = 4.
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                else: # 2^2 = 4 = a^2.
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "$ "

                if not seed[11]:
                    coq += "exists " + nat_to_token(base) + ". "
            else: # 4 = 2^2.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                if np.random.binomial(1, 0.5): # 4 = 2^2.
                    latex += nat_to_token(power)
                    latex += "= "
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                else: # 2^2 = 4.
                    latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                    latex += "= "
                    latex += nat_to_token(power)
                latex += "$ "

                if not seed[11]:
                    coq += "exists " + nat_to_token(base) + ". "
            if seed[16] and seed[6] < 2 and not seed[17]: # Prove bounds after power.
                if not (seed[11] and seed[23]):
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                else:
                    latex += np.random.choice(grammar.since + [""]).capitalize()
                latex += "$ "
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    if seed[11] and switch:
                        latex += var_to_token(vars[0])
                        latex += "= "
                        latex += nat_to_token(base)
                    else:
                        latex += nat_to_token(base)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(base)
                    coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar._is)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
            latex += ". "
    else: # Just prove 4 = 2^2.
        if seed[16] and seed[6] < 2 and seed[17]: # Prove bounds before power.
            latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
            if seed[6] == 0: # >= 2.
                if seed[11] and switch:
                    latex += var_to_token(vars[0])
                    latex += "= "
                    latex += nat_to_token(base)
                else:
                    latex += nat_to_token(base)

                coq += "assert ( <genH> : "
                coq += nat_to_token(base)
                coq += ">= <nat:2> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                if seed[11] and switch:
                    latex += var_to_token(vars[0])
                    latex += "= "
                    latex += nat_to_token(base)
                else:
                    latex += nat_to_token(base)

                coq += "assert ( <genH> : "
                coq += nat_to_token(base)
                coq += "> <nat:1> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += ". "
        
        if seed[11] and seed[23]: # No equations.
            latex += ""
        elif seed[11] and seed[21]: # 4 = a^2, no constants.
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            if np.random.binomial(1, 0.5): # 4 = a^2.
                latex += nat_to_token(power)
                latex += "= "
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            else: # a^2 = 4.
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(power)
            latex += "$ . "
        elif not seed[11] and seed[22]: # 4 = 2^2 -> a = 2.
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            if np.random.binomial(1, 0.5): # 4 = a^2.
                latex += nat_to_token(power)
                latex += "= "
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            else: # a^2 = 4.
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(power)
            latex += "$ "
            latex += np.random.choice(grammar.gives)
            latex += "$ " + var_to_token(vars[0]) + "= " + nat_to_token(base)

            coq += "exists " + nat_to_token(base) + ". "
            latex += "$ . "
        elif seed[20]: # 4 = a^2 = 2^2.
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            switch = np.random.randint(6)
            if switch == 0: # 4 = a^2 = 2^2.
                latex += nat_to_token(power)
                latex += "= "
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
            elif switch == 1: # 4 = 2^2 = a^2.
                latex += nat_to_token(power)
                latex += "= "
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            elif switch == 2: # a^2 = 2^2 = 4.
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(power)
            elif switch == 3: # a^2 = 4 = 2^2.
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(power)
                latex += "= "
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
            elif switch == 4: # 2^2 = a^2 = 4.
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(power)
            else: # 2^2 = 4 = a^2.
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(power)
                latex += "= "
                latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            latex += "$ . "

            if not seed[11]:
                coq += "exists " + nat_to_token(base) + ". "
        else: # 6 = 2(3).
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            if np.random.binomial(1, 0.5): # 4 = 2^2.
                latex += nat_to_token(power)
                latex += "= "
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
            else: # 2^2 = 4.
                latex += nat_to_token(base) + "^ " + nat_to_token(seed[0])
                latex += "= "
                latex += nat_to_token(power)
            latex += "$ . "

            if not seed[11]:
                coq += "exists " + nat_to_token(base) + ". "

        if seed[16] and seed[6] < 2 and not seed[17]: # Prove bounds after power.
            latex += np.random.choice(grammar.further).capitalize()
            latex += np.random.choice(grammar.note)
            latex += "$ "
            switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
            if seed[6] == 0: # >= 2.
                if seed[11] and switch:
                    latex += var_to_token(vars[0])
                    latex += "= "
                    latex += nat_to_token(base)
                else:
                    latex += nat_to_token(base)

                coq += "assert ( <genH> : "
                coq += nat_to_token(base)
                coq += ">= <nat:2> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                if seed[11] and switch:
                    latex += var_to_token(vars[0])
                    latex += "= "
                    latex += nat_to_token(base)
                else:
                    latex += nat_to_token(base)

                coq += "assert ( <genH> : "
                coq += nat_to_token(base)
                coq += "> <nat:1> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += ". "

    if seed[12] and not seed[14] and seed[13] and seed[1]:
        latex += np.random.choice(grammar.therefore).capitalize()
        latex += nat_to_token(power)
        latex += np.random.choice(grammar._is)
        latex += "a <def:" + defs[seed[0]] + "> "
        latex += np.random.choice(grammar.number)
        latex += np.random.choice(grammar.proven)
        latex += ". "
    if seed[12] and not seed[14]: # Restate theorem as proven.
        latex += np.random.choice(grammar.therefore).capitalize()
        if seed[15] and seed[6] < 2 and seed[7]: # Include bounds before restatement.
            latex += "$ "
            if seed[6] == 0: # >= 2.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += np.random.choice(grammar._and)
        
        switch = np.random.randint(3)
        if switch == 0: # 4 = a^2.
            latex += "$ "
            latex += nat_to_token(power)
            latex += "= "
            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            latex += "$ "
        elif switch == 1: # a^2 = 4.
            latex += "$ "
            latex += var_to_token(vars[0]) + "^ " + nat_to_token(seed[0])
            latex += "= "
            latex += nat_to_token(power)
            latex += "$ "
        else: # 4 is power.
            latex += nat_to_token(power)
            latex += np.random.choice(grammar._is)
            latex += np.random.choice(grammar.exponent)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5): # a in Z.
                latex += "$ "
                latex += var_to_token(vars[0])
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            else: # Number a.
                latex += np.random.choice(grammar.some)
                latex += np.random.choice(grammar.number)
                latex += "$ "
                latex += var_to_token(vars[0])
                latex += "$ "
            latex += np.random.choice(grammar.raised)
            latex += np.random.choice(grammar.power)
            latex += nat_to_token(seed[0])

        if seed[15] and seed[6] < 2 and not seed[7]: # Include bounds after restatement.
            latex += np.random.choice(grammar._and)
            latex += "$ "
            if seed[6] == 0: # >= 2.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                latex += var_to_token(vars[0])
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar._is)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
        latex += np.random.choice(grammar.proven)
        latex += ". "

    if seed[16] and seed[6] < 2: # Rewrites w/ assertions.
        coq += "split . "
        if (seed[1] and seed[7]) or not (seed[1] or seed[2]):
            coq += "apply <genH> . auto . "
        else:
            coq += "auto . apply <genH> . "
    elif seed[6] < 2:
        coq += "split . "
        if (seed[1] and seed[7]) or not (seed[1] or seed[2]):
            coq += "lia . auto . "
        else:
            coq += "auto . lia . "
    else:
        coq += "auto . "

    latex += r"\end{proof} \end{document} "
    coq += "Qed . "

    # Print.
    filename = args.path + "/" + args.set + "/powers_" + str(count) + "_" + str(seed[0]) + "_tokenized.txt"

    if args.debug:
        detok_coq = detokenize_coq(coq, "powers")
        proof, _ = check_proof(detok_coq, "")

        print("Example " + str(count) + ": ", end="")
        print(seed)
        print(latex)
        print(detok_coq)
        print("")
        if not proof:
            input("Invalid proof.")
    else:
        doc = open(filename, "w")
        doc.write(latex + "\n~\n" + coq)
        doc.close()

def composites(count):
    grammar = Composites()

    num_options = {
        "training": [2, 3, 5, 7, 9],
        "validation": [2, 3, 4, 5, 7, 8, 9],
        "test": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }

    seed = [
        np.random.choice(num_options[args.set]),    #  0: Number of factors.
        np.random.binomial(1, 0.5),                 #  1: Definition.
        np.random.binomial(1, 0.5),                 #  2: Define term at beginning of definition.
        np.random.binomial(1, 0.5),                 #  3: Set notation in theorem.
        np.random.binomial(1, 0.5),                 #  4: Logic notation (\exists).
        np.random.binomial(1, 0.5),                 #  5: Variable in theorem.
        np.random.randint(3),                       #  6: Lower bound condition. (0) >= 2, (1) > 1, (2) none.
        np.random.binomial(1, 0.5),                 #  7: If [6], 
        np.random.binomial(1, 0.5),                 #  8: If not [2], definition before name.
        np.random.binomial(1, 0.5),                 #  9: Let statement in proof.
        np.random.binomial(1, 0.5),                 # 10: Restatement of theorem.
        np.random.binomial(1, 0.5),                 # 11: If [1], restatement of definition.
        np.random.binomial(1, 0.5),                 # 12: If [10], state at (1) start or (0) end of proof.
        np.random.binomial(1, 0.5),                 # 13: If [6] and [10], include restatement of lower bound.
        np.random.binomial(1, 0.5),                 # 14: If [6], prove lower bound.
        np.random.binomial(1, 0.5),                 # 15: If [6] and [14], prove lower bound (1) before or (0) after factor proof.
        np.random.binomial(1, 0.5),                 # 16: Directional reasoning.
        np.random.binomial(1, 0.5),                 # 17: If [16], then (1) forward (ab = x -> th'm) or (0) backward (th'm, b.c. ab = x).
        np.random.binomial(1, 0.5),                 # 18: Intermediate algebraic steps. i.e. ab = 2(3) = 6.
        np.random.binomial(1, 0.5),                 # 19: If [9], don't use constants. i.e. 6 = ab not 6 = 2(3).
        np.random.binomial(1, 0.5),                 # 20: If not [9], factors imply variable values. i.e. 6 = 2(3) -> a = 2, b = 3.
        np.random.binomial(1, 0.5),                 # 21: If [9], no equations.
        np.random.binomial(1, 0.5),                 # 22: If [1], include definition in theorem.
        np.random.binomial(1, 0.5),                 # 23: If [1], have "x = 6" in theorem.
    ]

    factors = np.arange(2, 14)
    np.random.shuffle(factors)
    factors = factors[:seed[0]]
    product = 1
    for x in factors:
        product *= x

    vars = np.array(list(string.ascii_letters))
    np.random.shuffle(vars)
    vars = vars[:seed[0] + 1]
    order = np.arange(seed[0])

    defs = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve"
    ]

    latex = r"\documentclass[12pt]{article} \usepackage{amsthm,amsfonts,amssymb} \newtheorem{theorem}{Theorem} "
    if seed[1] and not seed[9]:
        latex += r"\newtheorem{definition}{Definition} "
    latex += r"\begin{document} "

    coq = "Require Import Lia . "

    np.random.shuffle(order)
    if seed[1]:
        coq += "Definition <def:" + defs[seed[0]] + "-composite> ( " + var_to_token(vars[-1]) + ": nat ) := "
        coq += "exists "
        for i in range(0, seed[0]):
            coq += var_to_token(vars[i])
        coq += ": nat , "

    # Definition
    if seed[1] and not seed[9]: # Explicit definition.
        latex += r"\begin{definition} "
        if seed[2]: # Let a, b at start of definition.
            if seed[3] and seed[4]: # Exists a, b in Z.
                if seed[6] < 2 and seed[7]:
                    latex += np.random.choice(grammar.exist).capitalize()
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                    latex += np.random.choice(grammar.exist)
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[3]: # Let a, b in Z.
                latex += np.random.choice(grammar.let).capitalize()
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[4]: # Exists a, b.
                if seed[6] < 2 and seed[7]:
                    latex += np.random.choice(grammar.exist[:-1])[:-2].capitalize()
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                    latex += np.random.choice(grammar.exist[:-1])[:-2]
                latex += np.random.choice(grammar.numbers)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
            else: # Let a, b.
                term = np.random.choice(grammar.let)
                latex += term.capitalize()
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += "$ "
                if term == "let ":
                    latex += grammar.be[0]
                else:
                    latex += np.random.choice(grammar.be[1:])
                latex += np.random.choice(grammar.numbers[:-1])
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
        else: # Let a, b later in definition.
            if seed[8]: # Definition before <def> token.
                latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.have)
                if seed[3]: # a, b in Z.
                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                else: # Let a, b.
                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
            else: # <def> token, then definition.
                if seed[3] and seed[4]: # Exists a, b in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exist)
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[3]: # Let a, b in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += "$ "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[4]: # Exists a, b.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exist[:-1])[:-2]
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.numbers)
                    latex += "$ "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                else: # Let a, b.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.numbers)
                    latex += "$ "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
        latex += r"\end{definition} "

    # Theorem
    latex += r"\begin{theorem} "
    if seed[1] and seed[9]: # Definition in theorem.
        if seed[2]: # Let a, b at start of definition.
            if seed[3] and seed[4]: # Exists a, b in Z.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += np.random.choice(grammar._if)
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.exist)
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[3]: # Let a, b in Z.
                latex += np.random.choice(grammar.let).capitalize()
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
            elif seed[4]: # Exists a, b.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += np.random.choice(grammar._if)
                else:
                    latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.exist[:-1])[:-2]
                latex += np.random.choice(grammar.numbers)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += "$ "
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += np.random.choice(grammar.then)
                    latex += np.random.choice(grammar.say)
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
            else: # Let a, b.
                term = np.random.choice(grammar.let)
                latex += term.capitalize()
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 1:
                        latex += "$ , $ "
                latex += "$ "
                if term == "let ":
                    latex += grammar.be[0]
                else:
                    latex += np.random.choice(grammar.be[1:])
                latex += np.random.choice(grammar.numbers[:-1])
                
                if seed[6] < 2 and seed[7]: # Bounds before product.
                    if seed[6] == 0: # >= 2.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        latex += ". "
                    else: # > 1.
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                        latex += ". "

                    # Do product.
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "/\ ( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
                elif seed[6] == 0: # >= 2.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.equal_to)
                            latex += np.random.choice(grammar.two)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + ">= <nat:2> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                    latex += ". "
                    coq += ". "
                elif seed[6] == 1: # > 1.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ "
                    coq += ") /\ "

                    # Do bounds.
                    latex += np.random.choice(grammar._and)
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.factors)
                        latex += "are "
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)

                        for i in range(seed[0]):
                            coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                            if i < seed[0] - 1:
                                coq += "/\ "
                    else:
                        if np.random.binomial(1, 0.5):
                            latex += np.random.choice(grammar.greater)
                            latex += np.random.choice(grammar.one)

                            for i in range(seed[0]):
                                coq += "( " + var_to_token(vars[i]) + "> <nat:1> ) "
                                if i < seed[0] - 1:
                                    coq += "/\ "
                        else:
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "
                    latex += ". "
                    coq += ". "
                else: # No bounds.
                    # Do product.
                    latex += ". "
                    latex += np.random.choice(grammar.say).capitalize()
                    if np.random.binomial(1, 0.5):
                        latex += "$ " + var_to_token(vars[-1]) + r"\in "
                        latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                        latex += "$ "
                    else:
                        latex += np.random.choice(grammar.some)
                        latex += np.random.choice(grammar.number)
                        latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar.said)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar.when)
                    
                    np.random.shuffle(order)
                    times = np.random.choice(grammar.times)

                    latex += "$ "
                    coq += "( "
                    if np.random.binomial(1, 0.5):
                        latex += var_to_token(vars[-1]) + "= "
                        coq += var_to_token(vars[-1]) + "= "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                    else:
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += var_to_token(vars[order[i]])
                            if i < seed[0] - 1:
                                latex += times
                                coq += "* "
                        latex += "= " + var_to_token(vars[-1])
                        coq += "= " + var_to_token(vars[-1])
                    latex += "$ . "
                    coq += ") . "
        else: # Let a, b later in definition.
            if seed[8]: # Definition before <def> token.
                latex += np.random.choice(grammar._if).capitalize()
                latex += np.random.choice(grammar.have)
                if seed[3]: # a, b in Z.
                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                else: # Let a, b.
                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar.such_that)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do definition.
                        if np.random.binomial(1, 0.5):
                            latex += ", "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.given)
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.numbers)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[i])
                                if i < seed[0] - 1:
                                    latex += "$ , $ "
                            latex += "$ , "
                            latex += np.random.choice(grammar.then)
                            latex += np.random.choice(grammar.say)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            latex += np.random.choice(grammar._is)
                            term = np.random.choice(grammar.number)
                            if term != "":
                                latex += "a "
                            latex += "<def:" + defs[seed[0]] + "-composite> "
                            latex += term
                        latex += ". "
                        coq += ". "
            else: # <def> token, then definition.
                if seed[3] and seed[4]: # Exists a, b in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exist)
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[3]: # Let a, b in Z.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += "$ "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += r"\in "
                    latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                    latex += "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                elif seed[4]: # Exists a, b.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.exist[:-1])[:-2]
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.numbers)
                    latex += "$ "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += "$ "
                    latex += np.random.choice(grammar.such_that)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "
                else: # Let a, b.
                    # Do definition.
                    latex += np.random.choice(grammar.say).capitalize()
                    latex += "$ " + var_to_token(vars[-1]) + "$ "
                    latex += np.random.choice(grammar._is)
                    term = np.random.choice(grammar.number)
                    if term != "":
                        latex += "a "
                    latex += "<def:" + defs[seed[0]] + "-composite> "
                    latex += term
                    latex += np.random.choice(grammar._if)
                    latex += np.random.choice(grammar.given)
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.numbers)
                    latex += "$ "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[i])
                        if i < seed[0] - 1:
                            latex += "$ , $ "
                    latex += "$ "
                    latex += np.random.choice(grammar.have)

                    if seed[6] < 2 and seed[7]: # Bounds before product.
                        if seed[6] == 0: # >= 2.
                            np.random.shuffle(order)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"\geq <nat:2> $ "
                        else: # > 1.
                            np.random.shuffle(order)
                            latex += np.random.choice(grammar.such_that)
                            latex += "$ "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                                if i < seed[0] - 2:
                                    latex += "$ , $ "
                                    coq += "/\ "
                                elif i < seed[0] - 1:
                                    latex += np.random.choice(grammar.listing)
                                    coq += "/\ "
                            latex += r"> <nat:1> $ "

                        # Do product.
                        latex += np.random.choice(grammar._and)
                        latex += np.random.choice(grammar.given)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ , "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ , "
                        
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "/\ ( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += ". "
                        coq += ". "
                    elif seed[6] == 0: # >= 2.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"\geq <nat:2> $ "
                        
                        latex += ". "
                        coq += ". "
                    elif seed[6] == 1: # > 1.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") /\ "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "

                        # Do bounds.
                        latex += np.random.choice(grammar._and)
                        np.random.shuffle(order)
                        latex += np.random.choice(grammar.such_that)
                        latex += "$ "
                        for i in range(seed[0]):
                            latex += var_to_token(vars[order[i]])
                            coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                            if i < seed[0] - 2:
                                latex += "$ , $ "
                                coq += "/\ "
                            elif i < seed[0] - 1:
                                latex += np.random.choice(grammar.listing)
                                coq += "/\ "
                        latex += r"> <nat:1> $ "
                        
                        latex += ". "
                        coq += ". "
                    else: # No bounds.
                        # Do product.
                        np.random.shuffle(order)
                        times = np.random.choice(grammar.times)

                        latex += "$ "
                        coq += "( "
                        if np.random.binomial(1, 0.5):
                            latex += var_to_token(vars[-1]) + "= "
                            coq += var_to_token(vars[-1]) + "= "
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                        else:
                            for i in range(seed[0]):
                                latex += var_to_token(vars[order[i]])
                                coq += var_to_token(vars[order[i]])
                                if i < seed[0] - 1:
                                    latex += times
                                    coq += "* "
                            latex += "= " + var_to_token(vars[-1])
                            coq += "= " + var_to_token(vars[-1])
                        latex += "$ "
                        coq += ") "

                        latex += np.random.choice(grammar._for)
                        if np.random.binomial(1, 0.5):
                            latex += "$ " + var_to_token(vars[-1]) + r"\in "
                            latex += np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                            latex += "$ "
                        else:
                            latex += np.random.choice(grammar.some)
                            latex += np.random.choice(grammar.number)
                            latex += "$ " + var_to_token(vars[-1]) + "$ "
                            
                        latex += ". "
                        coq += ". "

        latex += np.random.choice(grammar.so).capitalize()
        if seed[10] and np.random.binomial(1, 0.5):
            latex += "$ "
            latex += var_to_token(vars[-1])
            latex += "= "
            latex += nat_to_token(product)
            latex += "$ "
        elif seed[10]:
            latex += "$ "
            latex += nat_to_token(product)
            latex += "= "
            latex += var_to_token(vars[-1])
            latex += "$ "
        else:
            term = np.random.choice(grammar.number)
            if term != "":
                latex += "the "
            latex += term
            latex += nat_to_token(product)
        latex += np.random.choice(grammar._is)
        term = np.random.choice(grammar.number)
        if term != "":
            latex += "a "
        latex += "<def:" + defs[seed[0]] + "-composite> "
        latex += term
        latex += ". "

        coq += "Theorem <genP:1> : <def:" + defs[seed[0]] + "-composite> " + nat_to_token(product) + ". "
    elif seed[1]: # Explicit definition.
        coq += "Theorem <genP:1> : <def:" + defs[seed[0]] + "-composite> " + nat_to_token(product) + ". "

        if seed[10] and np.random.binomial(1, 0.5):
            latex += "$ "
            latex += var_to_token(vars[-1])
            latex += "= "
            latex += nat_to_token(product)
            latex += "$ "
        elif seed[10]:
            latex += "$ "
            latex += nat_to_token(product)
            latex += "= "
            latex += var_to_token(vars[-1])
            latex += "$ "
        else:
            term = np.random.choice(grammar.number)
            if term != "":
                latex += "The "
            latex += term
            latex += nat_to_token(product)
        latex += np.random.choice(grammar._is)
        term = np.random.choice(grammar.number)
        if term != "":
            latex += "a "
        latex += "<def:" + defs[seed[0]] + "-composite> "
        latex += term
        latex += ". "
    else: # No definition.
        coq += "Theorem <genP:1> : exists "
        for i in range(seed[0]):
            coq += var_to_token(vars[i])
        coq += ": nat , "

        if seed[2]: # 6 = ab for some ...
            if seed[10] and np.random.binomial(1, 0.5):
                latex += "$ "
                coq += "( "

                times = np.random.choice(grammar.times)
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                        coq += "* "
                latex += np.random.choice(grammar.equal)
                latex += nat_to_token(product)
                latex += "$ "

                coq += "= " + nat_to_token(product) + ") "
            elif seed[10]:
                latex += "$ "
                latex += nat_to_token(product)
                latex += np.random.choice(grammar.equal)
                coq += "( " + nat_to_token(product) + "= "

                times = np.random.choice(grammar.times)
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                        coq += "* "
                latex += "$ "
                coq += ") "
            else:
                term = np.random.choice(grammar.number)
                if term != "":
                    latex += "The "
                latex += term
                latex += nat_to_token(product)
                latex += np.random.choice(grammar._is)
                latex += np.random.choice(grammar.product)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.numbers)

                coq += "( " + nat_to_token(product) + "= "
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    coq += var_to_token(vars[i])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                    if i < seed[0] - 1:
                        coq += "* "
                if seed[3]:
                    latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                coq += ") "

            if seed[10] and seed[5]: # For some a, b
                latex += np.random.choice(grammar._for)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if seed[3]:
                    latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "

            if seed[6] == 0:
                latex += np.random.choice(grammar.such_that)
                np.random.shuffle(order)
                latex += "$ "
                coq += "/\ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                    if i < seed[0] - 1:
                        coq += "/\ "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            elif seed[6] == 1:
                latex += np.random.choice(grammar.such_that)
                np.random.shuffle(order)
                latex += "$ "
                coq += "/\ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                    if i < seed[0] - 1:
                        coq += "/\ "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += ". "
            coq += ". "
        else: # For some ... 6 = ab.
            if seed[3] and seed[4]:
                latex += np.random.choice(grammar.exist).capitalize()
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            elif seed[3]:
                latex += np.random.choice(grammar.let).capitalize()
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            elif seed[4]:
                term = np.random.choice(grammar.exist)
                if "there" in term:
                    latex += term[:-2].capitalize()
                    latex += np.random.choice(grammar.some)
                    latex += np.random.choice(grammar.numbers)
                    latex += "$ "
                else:
                    latex += term.capitalize()
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += "$ "
            else:
                term = np.random.choice(grammar.let)
                latex += term.capitalize()
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += "$ "
                if term == "let ":
                    latex += grammar.be[0]
                else:
                    latex += np.random.choice(grammar.be[1:])
                latex += np.random.choice(grammar.numbers[:-1])

            if seed[6] == 0:
                latex += np.random.choice(grammar.such_that)
                np.random.shuffle(order)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += "( " + var_to_token(vars[order[i]]) + ">= <nat:2> ) "
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                    coq += "/\ "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            elif seed[6] == 1:
                latex += np.random.choice(grammar.such_that)
                np.random.shuffle(order)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += "( " + var_to_token(vars[order[i]]) + "> <nat:1> ) "
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                    coq += "/\ "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            
            if seed[4]:
                if seed[6] < 2:
                    latex += np.random.choice(grammar._and)
                latex += np.random.choice(grammar.such_that)
            else:
                term = np.random.choice(grammar.such_that + [". "])
                if seed[6] < 2 and term != ". ":
                    latex += np.random.choice(grammar._and)
                    latex += term
                else:
                    latex += term

            if seed[10] and np.random.binomial(1, 0.5):
                latex += "$ "
                coq += "( "

                times = np.random.choice(grammar.times)
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                        coq += "* "
                latex += np.random.choice(grammar.equal)
                latex += nat_to_token(product)
                latex += "$ "

                coq += "= " + nat_to_token(product) + ") "
            elif seed[10]:
                latex += "$ "
                latex += nat_to_token(product)
                latex += np.random.choice(grammar.equal)
                coq += "( " + nat_to_token(product) + "= "

                times = np.random.choice(grammar.times)
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    coq += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                        coq += "* "
                latex += "$ "
                coq += ") "
            else:
                term = np.random.choice(grammar.number)
                if term != "":
                    latex += "the "
                latex += term
                latex += nat_to_token(product)
                latex += np.random.choice(grammar._is)
                latex += np.random.choice(grammar.product)
                latex += np.random.choice(grammar.of)
                latex += np.random.choice(grammar.numbers)

                coq += "( " + nat_to_token(product) + "= "
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[i])
                    coq += var_to_token(vars[i])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                    if i < seed[0] - 1:
                        coq += "* "
                if seed[3]:
                    latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
                coq += ") "
            latex += ". "
            coq += ". "
    latex += r"\end{theorem} "

    # Proof
    latex += r"\begin{proof} "
    coq += "Proof . "

    if seed[1]:
        coq += "unfold <def:" + defs[seed[0]] + "-composite> . " 

    if seed[13] and seed[1]: # Restate definition.
        latex += np.random.choice(grammar.recall).capitalize()
        if np.random.binomial(1, 0.5):
            latex += "a <def:" + defs[seed[0]] + "-composite> "
            latex += np.random.choice(grammar.number)
        else:
            latex += nat_to_token(product)
            latex += np.random.choice(grammar._is)
            term = np.random.choice(grammar.number)
            if term != "":
                latex += "a "
            latex += "<def:" + defs[seed[0]] + "-composite> "
            latex += term
            latex += np.random.choice(grammar._if)
            latex += np.random.choice(["it ", nat_to_token(product)])
        latex += np.random.choice(grammar._is)
        latex += np.random.choice(grammar.product)
        latex += np.random.choice(grammar.of)
        latex += np.random.choice(grammar.numbers)
        np.random.shuffle(order)
        latex += "$ "
        for i in range(seed[0]):
            latex += var_to_token(vars[order[i]])
            if i < seed[0] - 2:
                latex += "$ , $ "
            elif i < seed[0] - 1:
                latex += np.random.choice(grammar.listing)
        latex += "$ "

        if seed[6] == 0: # >= 2.
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.greater)
                latex += np.random.choice(grammar.equal_to)
                latex += np.random.choice(grammar.two)
            else:
                np.random.shuffle(order)
                latex += np.random.choice(grammar.such_that)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += r"\geq <nat:2> $ "
        else: # > 1.
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.greater)
                latex += np.random.choice(grammar.one)
            else:
                np.random.shuffle(order)
                latex += np.random.choice(grammar.such_that)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += r"> <nat:1> $ "
        latex += ". "

    if seed[11]: # Let a = 2, b = 3.
        latex += np.random.choice(grammar.let).capitalize()
        np.random.shuffle(order)
        latex += "$ "
        for i in range(seed[0]):
            latex += var_to_token(vars[order[i]])
            latex += "= "
            latex += nat_to_token(factors[order[i]])
            coq += "exists " + nat_to_token(factors[i]) + ". "
            if i < seed[0] - 2:
                latex += "$ , $ "
            elif i < seed[0] - 1:
                latex += np.random.choice(grammar.listing)
        latex += "$ . "

    if seed[12] and seed[14] and seed[13] and seed[1]: # Restate theorem to be proven, just tag.
        latex += np.random.choice(grammar.show).capitalize()
        latex += nat_to_token(product)
        latex += np.random.choice(grammar._is)
        term = np.random.choice(grammar.number)
        if term != "":
            latex += "a "
        latex += "<def:" + defs[seed[0]] + "-composite> "
        latex += term
        latex += ". "
    elif seed[12] and seed[14]: # Restate theorem, no tag.
        latex += np.random.choice(grammar.show).capitalize()
        if seed[15] and seed[6] < 2 and seed[7]: # Include bounds before restatement.
            np.random.shuffle(order)
            latex += "$ "
            if seed[6] == 0: # >= 2.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += np.random.choice(grammar._and)
        
        switch = np.random.randint(3)
        if switch == 0: # 6 = ab.
            latex += "$ "
            latex += nat_to_token(product)
            latex += "= "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            for i in range(seed[0]):
                latex += var_to_token(vars[order[i]])
                if i < seed[0] - 1:
                    latex += times
            latex += "$ "
        elif switch == 1: # ab = 6.
            latex += "$ "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            for i in range(seed[0]):
                latex += var_to_token(vars[order[i]])
                if i < seed[0] - 1:
                    latex += times
            latex += "= "
            latex += nat_to_token(product)
            latex += "$ "
        else: # 6 is product.
            latex += nat_to_token(product)
            latex += np.random.choice(grammar._is)
            latex += np.random.choice(grammar.product)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5): # a, b in Z.
                latex += "$ "
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            else: # Numbers a, b.
                latex += np.random.choice(grammar.some)
                latex += np.random.choice(grammar.numbers)
                latex += "$ "
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += "$ "

        if seed[15] and seed[6] < 2 and not seed[7]: # Include bounds after restatement.
            latex += np.random.choice(grammar._and)
            np.random.shuffle(order)
            latex += "$ "
            if seed[6] == 0: # >= 2.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
        latex += ". "

    if seed[18]: # Directional reasoning. i.e. 6 = 2(3) -> th'm.
        if seed[19]: # 6 = 2(3) -> th'm.
            if seed[16] and seed[6] < 2 and seed[17]: # Prove bounds before product.
                latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                np.random.shuffle(order)
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
                latex += ". "
            
            if seed[11] and seed[23]: # No equations.
                latex += ""
            elif seed[11] and seed[21]: # 6 = ab, no constants.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                if np.random.binomial(1, 0.5): # 6 = ab.
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                else: # ab = 6.
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                latex += "$ . "
            elif not seed[11] and seed[22]: # 6 = 2(3) -> a = 2, b = 3.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                if np.random.binomial(1, 0.5): # 6 = ab.
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                else: # ab = 6.
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                latex += "$ "
                latex += np.random.choice(grammar.gives)
                np.random.shuffle(order)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    latex += "= "
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)

                    coq += "exists " + nat_to_token(factors[i]) + ". "
                latex += "$ . "
            elif seed[20]: # 6 = ab = 2(3).
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                switch = np.random.randint(6)
                if switch == 0: # 6 = ab = 2(3).
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                elif switch == 1: # 6 = 2(3) = ab.
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                elif switch == 2: # ab = 2(3) = 6.
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                elif switch == 3: # ab = 6 = 2(3).
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                elif switch == 4: # 2(3) = ab = 6.
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                else: # 2(3) = 6 = ab.
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                latex += "$ . "

                if not seed[11]:
                    for i in range(seed[0]):
                        coq += "exists " + nat_to_token(factors[i]) + ". "
            else: # 6 = 2(3).
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar.further).capitalize()
                    latex += np.random.choice(grammar.note)
                    latex += np.random.choice(grammar.also)
                else:
                    latex += np.random.choice(grammar.note).capitalize()
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                if np.random.binomial(1, 0.5): # 6 = 2(3).
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                else: # 2(3) = 6.
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                latex += "$ . "

                if not seed[11]:
                    for i in range(seed[0]):
                        coq += "exists " + nat_to_token(factors[i]) + ". "
            if seed[16] and seed[6] < 2 and not seed[17]: # Prove bounds after product.
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += "$ "
                np.random.shuffle(order)
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
                latex += ". "

            latex += np.random.choice(grammar.this).capitalize()
            latex += np.random.choice(grammar.gives)
            if seed[1] and np.random.binomial(1, 0.5): # 6 is composite.
                latex += nat_to_token(product)
                latex += np.random.choice(grammar._is)
                term = np.random.choice(grammar.number)
                if term != "":
                    latex += "a "
                latex += "<def:" + defs[seed[0]] + "-composite> "
                latex += term
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += ". "
        else: # th'm b/c 6 = 2(3)
            latex += np.random.choice(grammar.prove).capitalize()
            if seed[1] and np.random.binomial(1, 0.5): # 6 is composite.
                latex += nat_to_token(product)
                latex += np.random.choice(grammar._is)
                term = np.random.choice(grammar.number)
                if term != "":
                    latex += "a "
                latex += "<def:" + defs[seed[0]] + "-composite> "
                latex += term
            else:
                latex += np.random.choice(grammar.our)
                latex += np.random.choice(grammar.theorem)
                latex += np.random.choice(grammar.holds)
            latex += np.random.choice(grammar.since)

            if seed[16] and seed[6] < 2 and seed[17]: # Prove bounds before product.
                latex += "$ "
                np.random.shuffle(order)
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
            
            if seed[11] and seed[23] and seed[16] and seed[6] < 2: # No equations.
                latex += ""
            elif seed[11] and seed[21]: # 6 = ab, no constants.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                if np.random.binomial(1, 0.5): # 6 = ab.
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                else: # ab = 6.
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                latex += "$ "
            elif not seed[11] and seed[22]: # 6 = 2(3) -> a = 2, b = 3.
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                if np.random.binomial(1, 0.5): # 6 = ab.
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                else: # ab = 6.
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                latex += "$ "
                latex += np.random.choice(grammar.gives)
                np.random.shuffle(order)
                latex += "$ "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    latex += "= "
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)

                    coq += "exists " + nat_to_token(factors[i]) + ". "
                latex += "$ "
            elif seed[20]: # 6 = ab = 2(3).
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                switch = np.random.randint(6)
                if switch == 0: # 6 = ab = 2(3).
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                elif switch == 1: # 6 = 2(3) = ab.
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                elif switch == 2: # ab = 2(3) = 6.
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                elif switch == 3: # ab = 6 = 2(3).
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                elif switch == 4: # 2(3) = ab = 6.
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                else: # 2(3) = 6 = ab.
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += var_to_token(vars[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                latex += "$ "

                if not seed[11]:
                    for i in range(seed[0]):
                        coq += "exists " + nat_to_token(factors[i]) + ". "
            else: # 6 = 2(3).
                if seed[16] and seed[6] < 2 and seed[17]:
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                latex += "$ "
                np.random.shuffle(order)
                times = np.random.choice(grammar.times)
                if np.random.binomial(1, 0.5): # 6 = 2(3).
                    latex += nat_to_token(product)
                    latex += "= "
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                else: # 2(3) = 6.
                    for i in range(seed[0]):
                        latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 1:
                            latex += times
                    latex += "= "
                    latex += nat_to_token(product)
                latex += "$ "

                if not seed[11]:
                    for i in range(seed[0]):
                        coq += "exists " + nat_to_token(factors[i]) + ". "
            if seed[16] and seed[6] < 2 and not seed[17]: # Prove bounds after product.
                if not (seed[11] and seed[23]):
                    latex += np.random.choice(grammar._and)
                    latex += np.random.choice(grammar.since + [""])
                else:
                    latex += np.random.choice(grammar.since + [""]).capitalize()
                latex += "$ "
                np.random.shuffle(order)
                switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
                if seed[6] == 0: # >= 2.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += ">= <nat:2> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"\geq <nat:2> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.equal_to)
                        latex += np.random.choice(grammar.two)
                else: # > 1.
                    for i in range(seed[0]):
                        if seed[11] and switch:
                            latex += var_to_token(vars[order[i]])
                            latex += "= "
                            latex += nat_to_token(factors[order[i]])
                        else:
                            latex += nat_to_token(factors[order[i]])
                        if i < seed[0] - 2:
                            latex += "$ , $ "
                        elif i < seed[0] - 1:
                            latex += np.random.choice(grammar.listing)

                        coq += "assert ( <genH> : "
                        coq += nat_to_token(factors[order[i]])
                        coq += "> <nat:1> ) . { lia . } "
                    if np.random.binomial(1, 0.5):
                        latex += r"> <nat:1> $ "
                    else:
                        latex += "$ "
                        latex += np.random.choice(grammar.are)
                        latex += np.random.choice(grammar.greater)
                        latex += np.random.choice(grammar.one)
            latex += ". "
    else: # Just prove 6 = 2(3).
        if seed[16] and seed[6] < 2 and seed[17]: # Prove bounds before product.
            latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            np.random.shuffle(order)
            switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
            if seed[6] == 0: # >= 2.
                for i in range(seed[0]):
                    if seed[11] and switch:
                        latex += var_to_token(vars[order[i]])
                        latex += "= "
                        latex += nat_to_token(factors[order[i]])
                    else:
                        latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(factors[order[i]])
                    coq += ">= <nat:2> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                for i in range(seed[0]):
                    if seed[11] and switch:
                        latex += var_to_token(vars[order[i]])
                        latex += "= "
                        latex += nat_to_token(factors[order[i]])
                    else:
                        latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(factors[order[i]])
                    coq += "> <nat:1> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += ". "
        
        if seed[11] and seed[23]: # No equations.
            latex += ""
        elif seed[11] and seed[21]: # 6 = ab, no constants.
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            if np.random.binomial(1, 0.5): # 6 = ab.
                latex += nat_to_token(product)
                latex += "= "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
            else: # ab = 6.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                latex += nat_to_token(product)
            latex += "$ . "
        elif not seed[11] and seed[22]: # 6 = 2(3) -> a = 2, b = 3.
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            if np.random.binomial(1, 0.5): # 6 = ab.
                latex += nat_to_token(product)
                latex += "= "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
            else: # ab = 6.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                latex += nat_to_token(product)
            latex += "$ "
            latex += np.random.choice(grammar.gives)
            np.random.shuffle(order)
            latex += "$ "
            for i in range(seed[0]):
                latex += var_to_token(vars[order[i]])
                latex += "= "
                latex += nat_to_token(factors[order[i]])
                if i < seed[0] - 2:
                    latex += "$ , $ "
                elif i < seed[0] - 1:
                    latex += np.random.choice(grammar.listing)

                coq += "exists " + nat_to_token(factors[i]) + ". "
            latex += "$ . "
        elif seed[20]: # 6 = ab = 2(3).
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            switch = np.random.randint(6)
            if switch == 0: # 6 = ab = 2(3).
                latex += nat_to_token(product)
                latex += "= "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
            elif switch == 1: # 6 = 2(3) = ab.
                latex += nat_to_token(product)
                latex += "= "
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
            elif switch == 2: # ab = 2(3) = 6.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                latex += nat_to_token(product)
            elif switch == 3: # ab = 6 = 2(3).
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                latex += nat_to_token(product)
                latex += "= "
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
            elif switch == 4: # 2(3) = ab = 6.
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                latex += nat_to_token(product)
            else: # 2(3) = 6 = ab.
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                latex += nat_to_token(product)
                latex += "= "
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 1:
                        latex += times
            latex += "$ . "

            if not seed[11]:
                for i in range(seed[0]):
                    coq += "exists " + nat_to_token(factors[i]) + ". "
        else: # 6 = 2(3).
            if seed[16] and seed[6] < 2 and seed[17]:
                latex += np.random.choice(grammar.further).capitalize()
                latex += np.random.choice(grammar.note)
                latex += np.random.choice(grammar.also)
            else:
                latex += np.random.choice(grammar.note).capitalize()
            latex += "$ "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            if np.random.binomial(1, 0.5): # 6 = 2(3).
                latex += nat_to_token(product)
                latex += "= "
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
            else: # 2(3) = 6.
                for i in range(seed[0]):
                    latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 1:
                        latex += times
                latex += "= "
                latex += nat_to_token(product)
            latex += "$ . "

            if not seed[11]:
                for i in range(seed[0]):
                    coq += "exists " + nat_to_token(factors[i]) + ". "

        if seed[16] and seed[6] < 2 and not seed[17]: # Prove bounds after product.
            latex += np.random.choice(grammar.further).capitalize()
            latex += np.random.choice(grammar.note)
            latex += "$ "
            np.random.shuffle(order)
            switch = np.random.binomial(1, 0.5) # a = 2 if seed[11].
            if seed[6] == 0: # >= 2.
                for i in range(seed[0]):
                    if seed[11] and switch:
                        latex += var_to_token(vars[order[i]])
                        latex += "= "
                        latex += nat_to_token(factors[order[i]])
                    else:
                        latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(factors[order[i]])
                    coq += ">= <nat:2> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                for i in range(seed[0]):
                    if seed[11] and switch:
                        latex += var_to_token(vars[order[i]])
                        latex += "= "
                        latex += nat_to_token(factors[order[i]])
                    else:
                        latex += nat_to_token(factors[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)

                    coq += "assert ( <genH> : "
                    coq += nat_to_token(factors[order[i]])
                    coq += "> <nat:1> ) . { lia . } "
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += ". "

    if seed[12] and not seed[14] and seed[13] and seed[1]:
        latex += np.random.choice(grammar.therefore).capitalize()
        latex += nat_to_token(product)
        latex += np.random.choice(grammar._is)
        term = np.random.choice(grammar.number)
        if term != "":
            latex += "a "
        latex += "<def:" + defs[seed[0]] + "-composite> "
        latex += term
        latex += np.random.choice(grammar.proven)
        latex += ". "
    if seed[12] and not seed[14]: # Restate theorem as proven.
        latex += np.random.choice(grammar.therefore).capitalize()
        if seed[15] and seed[6] < 2 and seed[7]: # Include bounds before restatement.
            np.random.shuffle(order)
            latex += "$ "
            if seed[6] == 0: # >= 2.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
            latex += np.random.choice(grammar._and)
        
        switch = np.random.randint(3)
        if switch == 0: # 6 = ab.
            latex += "$ "
            latex += nat_to_token(product)
            latex += "= "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            for i in range(seed[0]):
                latex += var_to_token(vars[order[i]])
                if i < seed[0] - 1:
                    latex += times
            latex += "$ "
        elif switch == 1: # ab = 6.
            latex += "$ "
            np.random.shuffle(order)
            times = np.random.choice(grammar.times)
            for i in range(seed[0]):
                latex += var_to_token(vars[order[i]])
                if i < seed[0] - 1:
                    latex += times
            latex += "= "
            latex += nat_to_token(product)
            latex += "$ "
        else: # 6 is product.
            latex += nat_to_token(product)
            latex += np.random.choice(grammar._is)
            latex += np.random.choice(grammar.product)
            latex += np.random.choice(grammar.of)
            if np.random.binomial(1, 0.5): # a, b in Z.
                latex += "$ "
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += r"\in " + np.random.choice([r"\mathbb{N} ", r"\mathbb{Z}^+ "])
                latex += "$ "
            else: # Numbers a, b.
                latex += np.random.choice(grammar.some)
                latex += np.random.choice(grammar.numbers)
                latex += "$ "
                np.random.shuffle(order)
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                latex += "$ "

        if seed[15] and seed[6] < 2 and not seed[7]: # Include bounds after restatement.
            latex += np.random.choice(grammar._and)
            np.random.shuffle(order)
            latex += "$ "
            if seed[6] == 0: # >= 2.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"\geq <nat:2> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.equal_to)
                    latex += np.random.choice(grammar.two)
            else: # > 1.
                for i in range(seed[0]):
                    latex += var_to_token(vars[order[i]])
                    if i < seed[0] - 2:
                        latex += "$ , $ "
                    elif i < seed[0] - 1:
                        latex += np.random.choice(grammar.listing)
                if np.random.binomial(1, 0.5):
                    latex += r"> <nat:1> $ "
                else:
                    latex += "$ "
                    latex += np.random.choice(grammar.are)
                    latex += np.random.choice(grammar.greater)
                    latex += np.random.choice(grammar.one)
        latex += np.random.choice(grammar.proven)
        latex += ". "

    if seed[16] and seed[6] < 2: # Rewrites w/ assertions.
        coq += "repeat split . "
        for i in range(seed[0]):
            coq += "all : try apply <genH> . "
    else:
        if seed[0] > 2:
            coq += "repeat split . all : "
        coq += "lia . "

    latex += r"\end{proof} \end{document} "
    coq += "Qed . "

    # Print.
    filename = args.path + "/" + args.set + "/composites_" + str(count) + "_" + str(seed[0]) + "_tokenized.txt"

    if args.debug:
        detok_coq = detokenize_coq(coq, "composites")
        proof, _ = check_proof(detok_coq, "")

        print("Example " + str(count) + ": ", end="")
        print(seed)
        print(latex)
        print(detok_coq)
        print("")
        if not proof:
            input("Invalid proof.")
    else:
        doc = open(filename, "w")
        doc.write(latex + "\n~\n" + coq)
        doc.close()

def sl1(d, coeff):
    l = 0
    for i in range(d + 1):
        if coeff[i] != 0:
            l += (i + 2)
    return l

def sl2(d, coeff):
    l = d
    for i in range(d + 1):
        if coeff[i] != 0:
            l += 2
    return l

def h(d):
    return d + 1

def program(method, d, coeff, vars):
    prog = []
    if method == 0:
        for i in range(d):
            if coeff[i] != 0:
                prog.append(var_to_token(vars[2] + str(i)) + ":= " + nat_to_token(coeff[i]) + "; ")
                for j in range(i):
                    prog.append(var_to_token(vars[2] + str(i)) + ":= " + var_to_token(vars[2] + str(i)) + "* " + var_to_token(vars[0]) + "; ")

        term = False
        for i in range(d - 1, -1, -1):
            if coeff[i] != 0:
                line = var_to_token(vars[1]) + ":= "
                if term:
                    line += var_to_token(vars[1]) + "+ "
                line += var_to_token(vars[2] + str(i))
                if i > 0:
                    line += "; "
                prog.append(line)
                term = True
    elif method == 1:
        for i in range(1, d):
            line = var_to_token(vars[2] + str(i)) + ":= "
            if i > 1:
                line += var_to_token(vars[2] + str(i - 1)) + "* "
            line += var_to_token(vars[0]) + "; "
            prog.append(line)

        for i in range(d):
            if coeff[i] != 0:
                line = var_to_token(vars[2] + str(i)) + ":= "
                if i > 0:
                    line += var_to_token(vars[2] + str(i)) + "* "
                line += nat_to_token(coeff[i]) + "; "
                prog.append(line)

        term = False
        for i in range(d - 1, -1, -1):
            if coeff[i] != 0:
                line = var_to_token(vars[1]) + ":= "
                if term:
                    line += var_to_token(vars[1]) + "+ "
                line += var_to_token(vars[2] + str(i))
                if i < 0:
                    line += "; "
                prog.append(line)
                term = True
    elif method == 2:
        for i in range(d - 1, -1, -1):
            line = var_to_token(vars[1]) + ":= "
            if coeff[i] != 0:
                line += nat_to_token(coeff[i])
                if i < d - 1:
                    line += "+ "
            if i < d - 1:
                line += var_to_token(vars[1]) + "* " + var_to_token(vars[0])
            if i > 0:
                line += "; "
            prog.append(line)

    return prog

def condition(method, d, coeff, vars, coq=False):
    post = []
    
    grammar = Straightline()
    land = "/\ " if coq else r"\land "
    times = "* " if coq else np.random.choice(grammar.times)
    
    if method == 0:
        for i in range(d):
            if coeff[i] != 0:
                for j in range(i + 1):
                    line = var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + land
                    for k in range(i):
                        if coeff[k] != 0:
                            line += var_to_token(vars[2] + str(k)) + "= " + nat_to_token(coeff[k])
                            if k == 1:
                                line += times + var_to_token(vars[3])
                            elif k > 0:
                                line += times + var_to_token(vars[3]) + "^ " + nat_to_token(k)
                            line += land
                    line += var_to_token(vars[2] + str(i)) + "= " + nat_to_token(coeff[i])
                    if j == 1:
                        line += times + var_to_token(vars[3])
                    elif j > 0:
                        line += times + var_to_token(vars[3]) + "^ " + nat_to_token(j)
                    post.append(line)

        for i in range(d - 1, -1, -1):
            if coeff[i] != 0:
                line = var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + land
                for j in range(i):
                    if coeff[j] != 0:
                        line += var_to_token(vars[2] + str(j)) + "= " + nat_to_token(coeff[j])
                        if j == 1:
                            line += times + var_to_token(vars[3])
                        elif j > 0:
                            line += times + var_to_token(vars[3]) + "^ " + nat_to_token(j)
                        line += land
                line += var_to_token(vars[1]) + "= "
                for j in range(d - 1, i - 1, -1):
                    if coeff[j] != 0:
                        line += nat_to_token(coeff[j])
                        if j == 1:
                            line += times + var_to_token(vars[3])
                        elif j > 0:
                            line += times + var_to_token(vars[3]) + "^ " + nat_to_token(j)

                        if j != i:
                            line += "+ "
                post.append(line)
    elif method == 1:
        for i in range(1, d):
            line = var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + land
            for j in range(1, i + 1):
                line += var_to_token(vars[2] + str(j)) + "= " + var_to_token(vars[3])
                if j > 1:
                    line += "^ " + nat_to_token(j)
                if j < i:
                    line += land
            post.append(line)

        for i in range(d):
            if coeff[i] != 0:
                line = var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + land
                for j in range(1, i + 1):
                    line += var_to_token(vars[2] + str(j)) + "= "
                    if coeff[j] != 0:
                        line += nat_to_token(coeff[j]) + times
                    line += var_to_token(vars[3])
                    if j > 1:
                        line += "^ " + nat_to_token(j)
                    line += land
                    
                for j in range(i + 1, d):
                    line += var_to_token(vars[2] + str(j)) + "= " + var_to_token(vars[3])
                    if j > 1:
                        line += "^ " + nat_to_token(j)
                    line += land

                line += var_to_token(vars[2] + str(0)) + "= " + nat_to_token(coeff[0])
                post.append(line)

        for i in range(d - 1, -1, -1):
            if coeff[i] != 0:
                line = var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + land
                for j in range(1, i):
                    line += var_to_token(vars[2] + str(j)) + "= "
                    if coeff[j] != 0:
                        line += nat_to_token(coeff[j]) + times
                    line += var_to_token(vars[3])
                    if j > 1:
                        line += "^ " + nat_to_token(j)
                    line += land
                if i > 0:
                    line += var_to_token(vars[2] + str(0)) + "= " + nat_to_token(coeff[0]) + land

                line += var_to_token(vars[1]) + "= "
                for j in range(d - 1, i - 1, -1):
                    if coeff[j] != 0:
                        line += nat_to_token(coeff[j])
                        if j == 1:
                            line += times + var_to_token(vars[3])
                        elif j > 0:
                            line += times + var_to_token(vars[3]) + "^ " + nat_to_token(j)
                        
                        if j > i:
                            line += "+ "
                post.append(line)
    elif method == 2:
        for i in range(d - 1, -1, -1):
            line = var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + land
            line += var_to_token(vars[1]) + "= "
            for j in range(d - 1, i - 1, -1):
                if coeff[j] != 0:
                    line += nat_to_token(coeff[j])
                    if j - i == 1:
                        line += times + var_to_token(vars[3])
                    elif j > i:
                        line += times + var_to_token(vars[3]) + "^ " + nat_to_token(j - i)
                    if j > i:
                        line += "+ "
            post.append(line)

    return post

def poly(count):
    grammar = Poly()

    num_options = {
        "training": [2, 3, 5, 7, 9, 11],
        "validation": [2, 3, 4, 5, 7, 8, 9], # not implemented
        "test": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    }

    seed = [
        np.random.randint(1, 20),                   #  0: Polynomial degree.
        np.random.binomial(1, 0.5),                 #  1: Print definition.
        np.random.randint(3),                       #  2: Polynomial evaluation method. 0: naive, 1: store powers, 2: Horner
        np.random.binomial(1, 0.5),                 #  3: Introduction for the program (if seed[1] == 1).
        np.random.binomial(1, 0.5),                 #  4: For all statement.
        np.random.binomial(1, 0.5),                 #  5: Position of for all statement (if seed[4] == 1).
        np.random.binomial(1, 0.5),                 #  6: State "after program completes".
        np.random.binomial(1, 0.5),                 #  7: Put program at start of theorem (if seed[1] == 0).
        np.random.binomial(1, 0.5),                 #  8: Introduction for the proof.
        np.random.binomial(1, 0.5),                 #  9: Mid-proof expositions.
        np.random.binomial(1, 0.5),                 # 10: Forward proof. (i.e. Hoare logic gives ... :. correctness is shown.)
        np.random.binomial(1, 0.5),                 # 11: Verbose proof (no eqnarray).
        np.random.binomial(1, 0.5),                 # 12: Hoare triples always (if seed[11] == 1).
        np.random.randint(4),                       # 13: Display for Hoare triples. 0: eqnarray, 1: verbatim, 2: tabular, 3: inline verb
    ]

    vars = np.array(list(string.ascii_letters))
    np.random.shuffle(vars)
    vars = vars[:4] # x, y, t, n

    coeff = [np.random.randint(0, 5) for i in range(seed[0] - 2)]
    coeff.insert(0, np.random.randint(1, 5))
    if seed[0] > 1:
        coeff.append(np.random.randint(1, 5))
    coeff = np.array(coeff)
    assert(len(coeff) == seed[0])

    lines = 0
    if seed[2] == 0:
        lines = sl1(seed[0] - 1, coeff)
    elif seed[2] == 1:
        lines = sl2(seed[0] - 1, coeff)
    elif seed[2] == 2:
        lines = h(seed[0] - 1)

    while lines not in num_options[args.set]:
        seed[0] = np.random.randint(1, 20)
        seed[2] = np.random.randint(3)

        coeff = [np.random.randint(0, 10) for i in range(seed[0] - 2)]
        coeff.insert(0, np.random.randint(1, 10))
        if seed[0] > 1:
            coeff.append(np.random.randint(1, 10))
        coeff = np.array(coeff)
        assert(len(coeff) == seed[0])

        if seed[2] == 0:
            lines = sl1(seed[0] - 1, coeff)
        elif seed[2] == 1:
            lines = sl2(seed[0] - 1, coeff)
        elif seed[3] == 2:
            lines = h(seed[0] - 1)

    prog = program(seed[2], seed[0], coeff, vars)

    latex_cond = condition(seed[2], seed[0], coeff, vars)
    coq_cond = condition(seed[2], seed[0], coeff, vars, coq=True)

    if args.debug:
        print(seed)
        print(prog)
        print(latex_cond)
        print(len(prog), len(latex_cond), lines)
        print(coeff)
    assert(len(prog) == lines)
    assert(len(latex_cond) == lines)
    assert(len(coq_cond) == lines)

    latex = r"\begin{document} "

    coq = "Require Import String . From PLF Require Import Imp . From PLF Require Import Hoare . "
    for i in range(seed[0]):
        if (coeff[i] != 0 or seed[2] == 1) and seed[2] != 2:
            coq += "Definition " + var_to_token(vars[2] + str(i)) + ": string := \" " + var_to_token(vars[2] + str(i)) + "\" . "

    # Theorem

    if seed[1]:
        latex += r"\begin{definition} "
        latex += np.random.choice(grammar.consider)
        latex += np.random.choice(grammar.program)
        latex += r"<def:poly> in \emph{Imp} "
        latex += np.random.choice(grammar.such)
        latex += r"\begin{verbatim} "

        coq += "Definition <def:poly> := "

        for line in prog:
            latex += line
            coq += line

        latex += r"\end{verbatim} \end{definition} \begin{theorem} "
        coq += ". "

        if seed[3]:
            latex += np.random.choice(grammar.consider)
            latex += np.random.choice(grammar.program)
            latex += "<def:poly> "
            latex += np.random.choice([r"in \emph{Imp} . ", ". "])
        
        latex += np.random.choice(grammar.assume)
        if seed[4] and seed[5]:
            separator = np.random.choice(["", ", ", "( ", "--- "])
            latex += separator

            latex += np.random.choice(grammar.forall)
            
            phrase = np.random.choice(grammar.natural)
            if phrase in [r"\in \mathbb{N} ", r"\in \mathbb{Z}^+ "]:
                latex += "$ " + var_to_token(vars[3]) + phrase + "$ "
            else:
                latex += phrase + "$ " + var_to_token(vars[3]) + "$ "

            if separator == "( ":
                latex += ") "
            else:
                latex += separator

        latex += np.random.choice(grammar.that)
        latex += "$ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + "$ "

        if seed[4] and not seed[5]:
            separator = np.random.choice(["", ", ", "( ", "--- "])
            latex += separator

            latex += np.random.choice(grammar.forall)
            
            phrase = np.random.choice(grammar.natural)
            if phrase in [r"\in \mathbb{N} ", r"\in \mathbb{Z}^+ "]:
                latex += "$ " + var_to_token(vars[3]) + phrase + "$ "
            else:
                latex += phrase + "$ " + var_to_token(vars[3]) + "$ "

            if separator == "( ":
                latex += ") "
            else:
                latex += separator

        latex += np.random.choice(grammar.prior)
        latex += np.random.choice(grammar.execution)
        latex += np.random.choice(grammar.this)
        latex += np.random.choice(grammar.program)
        latex += np.random.choice(["", ", "])
        latex += np.random.choice(grammar.then)
        latex += np.random.choice(grammar.have)

        latex += "$ " + var_to_token(vars[1]) + "= "
        times = np.random.choice(grammar.times)
        for i in range(seed[0] - 1, -1, -1):
            if coeff[i] != 0:
                latex += nat_to_token(coeff[i])
                if i == 1:
                    latex += times + var_to_token(vars[3]) + "+ "
                elif i > 0:
                    latex += times + var_to_token(vars[3]) + "^ " + nat_to_token(i) + "+ "
        latex += "$ "
        
        if seed[6]:
            latex += np.random.choice(grammar.after)
            latex += np.random.choice(grammar.program)
            latex += np.random.choice(grammar.done)
        latex += r". \end{theorem} "

        coq += "Theorem <genH:poly_code_correct> : forall "
        coq += var_to_token(vars[3])
        coq += r": nat , {{ "
        coq += var_to_token(vars[0])
        coq += "= "
        coq += var_to_token(vars[3])
        coq += r"}} <def:poly> {{ "
        coq += var_to_token(vars[1])
        coq += "= "
        for i in range(seed[0] - 1, -1, -1):
            if coeff[i] != 0:
                coq += nat_to_token(coeff[i])
                if i == 1:
                    coq += "* " + var_to_token(vars[3]) + "+ "
                elif i > 0:
                    coq += "* " + var_to_token(vars[3]) + "^ " + nat_to_token(i) + "+ "
        coq += r"}} . "
    else:
        latex += r"\begin{theorem} "
        latex += np.random.choice(grammar.consider)
        latex += np.random.choice(grammar.program)
        latex += np.random.choice([r"in \emph{Imp} ", ""])
        latex += np.random.choice(grammar.such)

        latex += r"\begin{verbatim} "
        for line in prog:
            latex += line
        latex += r"\end{verbatim} "
        
        latex += np.random.choice(grammar.assume)
        if seed[4] and seed[5]:
            separator = np.random.choice(["", ", ", "( ", "--- "])
            latex += separator

            latex += np.random.choice(grammar.forall)
            
            phrase = np.random.choice(grammar.natural)
            if phrase in [r"\in \mathbb{N} ", r"\in \mathbb{Z}^+ "]:
                latex += "$ " + var_to_token(vars[3]) + phrase + "$ "
            else:
                latex += phrase + "$ " + var_to_token(vars[3]) + "$ "

            if separator == "( ":
                latex += ") "
            else:
                latex += separator

        latex += np.random.choice(grammar.that)
        latex += "$ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + "$ "

        if seed[4] and not seed[5]:
            separator = np.random.choice(["", ", ", "( ", "--- "])
            latex += separator

            latex += np.random.choice(grammar.forall)
            
            phrase = np.random.choice(grammar.natural)
            if phrase in [r"\in \mathbb{N} ", r"\in \mathbb{Z}^+ "]:
                latex += "$ " + var_to_token(vars[3]) + phrase + "$ "
            else:
                latex += phrase + "$ " + var_to_token(vars[3]) + "$ "

            if separator == "( ":
                latex += ") "
            else:
                latex += separator

        latex += np.random.choice(grammar.prior)
        latex += np.random.choice(grammar.execution)
        latex += np.random.choice(grammar.this)
        latex += np.random.choice(grammar.program)
        latex += np.random.choice(["", ", "])
        latex += np.random.choice(grammar.then)
        latex += np.random.choice(grammar.have)

        latex += "$ " + var_to_token(vars[1]) + "= "
        times = np.random.choice(grammar.times)
        for i in range(seed[0] - 1, -1, -1):
            if coeff[i] != 0:
                latex += nat_to_token(coeff[i])
                if i == 1:
                    latex += times + var_to_token(vars[3]) + "+ "
                elif i > 0:
                    latex += times + var_to_token(vars[3]) + "^ " + nat_to_token(i) + "+ "
        latex += "$ "
        
        if seed[6]:
            latex += np.random.choice(grammar.after)
            latex += np.random.choice(grammar.program)
            latex += np.random.choice(grammar.done)
        latex += r". \end{theorem} "

        coq += "Theorem <genH:poly_code_correct> : forall "
        coq += var_to_token(vars[3])
        coq += r": nat , {{ "
        coq += var_to_token(vars[0])
        coq += "= "
        coq += var_to_token(vars[3])
        coq += r"}} "

        for line in prog:
            coq += line

        coq += r"{{ "
        coq += var_to_token(vars[1])
        coq += "= "
        for i in range(seed[0] - 1, -1, -1):
            if coeff[i] != 0:
                coq += nat_to_token(coeff[i])
                if i == 1:
                    coq += "* " + var_to_token(vars[3]) + "+ "
                elif i > 0:
                    coq += "* " + var_to_token(vars[3]) + "^ " + nat_to_token(i) + "+ "
        coq += r"}} . "

    # Proof
    
    latex += r"\begin{proof} "
    coq += "Proof . intros . "

    if seed[11]:
        # Introduce the proof. We have to mention Hoare logic.
        latex += np.random.choice(grammar.the_proof)
        latex += np.random.choice(grammar.this)
        latex += np.random.choice(grammar.program)
        latex += np.random.choice(grammar.demonstrated)
        latex += np.random.choice(grammar.by)
        latex += np.random.choice(grammar.hoare)
        latex += np.random.choice(grammar.logic)
        latex += ". "

        # Loop through the program and make statements about it.
        # Sometimes combine steps.
        i = 0
        while i < lines:
            if seed[12] == 0:
                l = np.random.randint(1, lines - i + 1)
                i += l

                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.considering).capitalize()
                    latex += np.random.choice(grammar.the_next)
                    latex += np.random.choice(grammar.step) if (l == 1) else np.random.choice(grammar.steps)
                    latex += np.random.choice(grammar.in_the)
                    latex += np.random.choice(grammar.program)
                else:
                    latex += np.random.choice(grammar.next).capitalize()
                latex += ", "
                latex += np.random.choice(grammar.have)
                
                # Jump into triple.
                if seed[13] == 0:   # \begin{eqnarray}
                    latex += r"\begin{eqnarray} "
                    for j in range(l):
                        latex += (r"\{ " + latex_cond[j - 1] + r"\} \, ") if (j > 0) else r"\{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + r"\} \, "
                        latex += prog[j][:-2] + r"\, "
                        latex += r"\{ " + latex_cond[j] + r"\} " + (r"\\ " if (j < l - 1) else "")
                    latex += r"\end{eqnarray} "
                elif seed[13] == 1: # \begin{verbatim}
                    latex += r"\begin{verbatim} "
                    for j in range(l):
                        latex += ("{ " + latex_cond[j - 1] + "} ") if (j > 0) else r"{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + "} "
                        latex += prog[j][:-2]
                        latex += "{ " + latex_cond[j] + "} "
                    latex += r"\end{verbatim} "
                elif seed[13] == 2: # \begin{tabular}
                    latex += r"\[ \begin{tabular} {rcl} "
                    for j in range(l):
                        latex += (r"\{ " + latex_cond[j - 1] + r"\} & ") if (j > 0) else r"\{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + r"\} & "
                        latex += prog[j][:-2] + "& "
                        latex += r"\{ " + latex_cond[j] + r"\} " + (r"\\ " if (j < l - 1) else "")
                    latex += r"\end{tabular} "
                elif seed[13] == 3: # \verb|...|
                    for j in range(l):
                        latex += r"\verb | "
                        latex += ("{ " + latex_cond[j - 1] + "} ") if (j > 0) else r"{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + "} "
                        latex += prog[j][:-2]
                        latex += "{ " + latex_cond[j] + "} "
                        latex += "| "

                        if l > 2 and j < l - 2:
                            latex += ", "
                        elif l > 1 and j < l - 1:
                            latex += "and "
                    latex += ". "
                
                # Mention Hoare assignment rule.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.given).capitalize()
                    latex += np.random.choice(grammar.by)
                    latex += "the " + np.random.choice(grammar.assignment)
                    latex += np.random.choice(grammar.rule)
                    latex += np.random.choice(grammar.of)
                    latex += np.random.choice(grammar.hoare)
                    latex += np.random.choice(grammar.logic)
                    latex += ". "
            elif seed[12] == 1:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.considering).capitalize()
                    latex += np.random.choice(grammar.the_next)
                    latex += np.random.choice(grammar.step)
                    latex += np.random.choice(grammar.in_the)
                    latex += np.random.choice(grammar.program)
                else:
                    latex += np.random.choice(grammar.next).capitalize()
                latex += ", "
                latex += np.random.choice(grammar.have)
                
                # Break down the command.
                latex += "the "
                latex += np.random.choice(grammar.assignment)
                latex += np.random.choice(grammar.command)
                # "Assigning 24 to Y" vs. "Y := 24."
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.assigning)
                    latex += "$ " + prog[i].split(" := ")[1][:-2] + "$ "
                    latex += np.random.choice(grammar.to)
                    latex += "$ " + prog[i].split(" := ")[0] + "$ . "
                else:
                    latex += r"\verb | " + prog[i] + "| . "

                latex += "This "
                if np.random.binomial(1, 0.5):
                    latex += ""
                else:
                    latex += np.random.choice(grammar.command)
                latex += np.random.choice(grammar.takes)
                latex += np.random.choice(grammar.precondition)
                latex += "$ " + ((r"\{ " + latex_cond[i - 1] + r"\} ") if (i > 0) else r"\{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + r"\} ") + "$ "
                latex += np.random.choice(grammar.to)
                latex += np.random.choice(grammar.postcondition)
                latex += r"$ \{ " + latex_cond[i] + r"\} $ . "
                
                # Mention Hoare assignment rule.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.given).capitalize()
                    latex += np.random.choice(grammar.by)
                    latex += "the assignment "
                    latex += np.random.choice(grammar.rule)
                    latex += np.random.choice(grammar.of)
                    latex += np.random.choice(grammar.hoare)
                    latex += np.random.choice(grammar.logic)
                    latex += ". "

                i += 1
            elif seed[12] == 2:
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.considering).capitalize()
                    latex += np.random.choice(grammar.the_next)
                    latex += np.random.choice(grammar.step) if (l == 1) else np.random.choice(grammar.steps)
                    latex += np.random.choice(grammar.in_the)
                    latex += np.random.choice(grammar.program)
                else:
                    latex += np.random.choice(grammar.next).capitalize()
                latex += ", "
                latex += np.random.choice(grammar.have)
                
                # Break down the command or jump into triple.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.assignment)
                    latex += np.random.choice(grammar.command)
                    # "Assigning 24 to Y" vs. "Y := 24."
                    if np.random.binomial(1, 0.5):
                        latex += np.random.choice(grammar.assigning)
                        latex += "$ " + prog[i].split(" := ")[1][:-2] + "$ "
                        latex += np.random.choice(grammar.to)
                        latex += "$ " + prog[i].split(" := ")[0] + "$ . "
                    else:
                        latex += r"\verb | " + prog[i] + "| . "

                    latex += "This "
                    if np.random.binomial(1, 0.5):
                        latex += ""
                    else:
                        latex += np.random.choice(grammar.command)
                    latex += np.random.choice(grammar.takes)
                    latex += np.random.choice(grammar.precondition)
                    latex += "$ " + ((r"\{ " + latex_cond[i - 1] + r"\} ") if (i > 0) else r"\{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + r"\} ") + "$ "
                    latex += np.random.choice(grammar.to)
                    latex += np.random.choice(grammar.postcondition)
                    latex += r"$ \{ " + latex_cond[i] + r"\} $ . "

                    i += 1
                else:
                    l = np.random.randint(1, lines - i + 1)
                    i += l
                    
                    if seed[13] == 0:   # \begin{eqnarray}
                        latex += r"\begin{eqnarray} "
                        for j in range(l):
                            latex += (r"\{ " + latex_cond[j - 1] + r"\} \, ") if (j > 0) else r"\{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + r"\} \, "
                            latex += prog[j][:-2] + r"\, "
                            latex += r"\{ " + latex_cond[j] + r"\} " + (r"\\ " if (j < l - 1) else "")
                        latex += r"\end{eqnarray} "
                    elif seed[13] == 1: # \begin{verbatim}
                        latex += r"\begin{verbatim} "
                        for j in range(l):
                            latex += ("{ " + latex_cond[j - 1] + "} ") if (j > 0) else r"{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + "} "
                            latex += prog[j][:-2]
                            latex += "{ " + latex_cond[j] + "} "
                        latex += r"\end{verbatim} "
                    elif seed[13] == 2: # \begin{tabular}
                        latex += r"\[ \begin{tabular} {rcl} "
                        for j in range(l):
                            latex += (r"\{ " + latex_cond[j - 1] + r"\} & ") if (j > 0) else r"\{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + r"\} & "
                            latex += prog[j][:-2] + "& "
                            latex += r"\{ " + latex_cond[j] + r"\} " + (r"\\ " if (j < l - 1) else "")
                        latex += r"\end{tabular} "
                    elif seed[13] == 3: # \verb|...|
                        for j in range(l):
                            latex += r"\verb | "
                            latex += ("{ " + latex_cond[j - 1][3:-3] + "} ") if (j > 0) else r"{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + "} "
                            latex += prog[j][:-2]
                            latex += "{ " + latex_cond[j][3:-3] + "} "
                            latex += "| "

                            if l > 2 and j < l - 2:
                                latex += ", "
                            elif l > 1 and j < l - 1:
                                latex += "and "
                        latex += ". "
                
                # Mention Hoare assignment rule.
                if np.random.binomial(1, 0.5):
                    latex += np.random.choice(grammar.given).capitalize()
                    latex += np.random.choice(grammar.by)
                    latex += "the assignment "
                    latex += np.random.choice(grammar.rule)
                    latex += np.random.choice(grammar.of)
                    latex += np.random.choice(grammar.hoare)
                    latex += np.random.choice(grammar.logic)
                    latex += ". "
    else:
        if not seed[10] and seed[8]:
            if np.random.binomial(1, 0.5):
                latex += np.random.choice(grammar.the_proof)
                latex += np.random.choice(grammar.this)
                latex += np.random.choice(grammar.program)
                latex += np.random.choice(grammar.demonstrated)
                latex += np.random.choice(grammar.by)
                latex += np.random.choice(grammar.following)
                latex += "decorated "
                latex += np.random.choice(grammar.program)
                latex += ". "
            else:
                latex += np.random.choice(grammar.observe)
                latex += np.random.choice(grammar.following)
                latex += "decorated "
                latex += np.random.choice(grammar.program)
                latex += ". "
        elif seed[10]:
            latex += np.random.choice(grammar.applying)
            latex += np.random.choice(grammar.hoare)
            latex += np.random.choice(grammar.logic)
            latex += np.random.choice(grammar.gives)
            latex += ": "
        
        times = np.random.choice(grammar.times)

        latex += r"\begin{eqnarray} "
        latex += r"\{ " + var_to_token(vars[0]) + "= " + var_to_token(vars[3]) + r"\} \\ "
        for i in range(lines):
            latex += prog[i] + r"\\ "
            latex += r"\{ " + latex_cond[i] + r"\} "

            if seed[9] and np.random.binomial(1, 0.05):
                latex += r"\end{eqnarray} "
                latex += np.random.choice(grammar.further)
                latex += np.random.choice(grammar.execution)
                latex += np.random.choice(grammar.this)
                latex += np.random.choice(grammar.program)
                latex += np.random.choice(grammar.gives)
                latex += ": "
                latex += r"\begin{eqnarray} "
            elif i < lines - 1:
                latex += r"\\ "
        
        latex += r"\end{eqnarray} "
        if seed[10]:
            latex += np.random.choice(grammar.therefore)
            latex += np.random.choice(grammar.this)
            latex += np.random.choice(grammar.program)
            latex += np.random.choice(grammar.proven)
            latex += "correct . "
        else:
            latex += np.random.choice(grammar.applying)
            latex += np.random.choice(grammar.hoare)
            latex += np.random.choice(grammar.logic)
            latex += np.random.choice(grammar.completes)
            latex += np.random.choice(grammar.proof)
            latex += ". "

    latex += r"\end{proof} \end{document} "

    for line in coq_cond:
        coq += "apply hoare_seq with ( Q := ( ( " + line + r") ) %assertion ) . "
    coq += "all : eapply hoare_consequence_pre ; try ( apply hoare_asgn || assn_auto'' ) . Qed . "

    # Print.
    filename = args.path + "/" + args.set + "/poly_" + str(count) + "_" + str(lines) + "_" + str(seed[2]) + "_tokenized.txt"

    if args.debug:
        detok_coq = detokenize_coq(coq, "poly")
        proof, _ = check_proof(detok_coq, "")

        print("Example " + str(count) + ": ", end="")
        print(seed)
        print(latex)
        print(detok_coq)
        print("")
        if not proof:
            input("Invalid proof.")
    else:
        doc = open(filename, "w")
        doc.write(latex + "\n~\n" + coq)
        doc.close()

def main():
    # Parse -t argument to generate dataset.
    # Then create the given -n examples.
    if args.type == "even-odd":
        for i in range(args.num):
            even_odd(i)
    elif args.type == "powers":
        for i in range(args.num):
            powers(i)
    elif args.type == "composites":
        for i in range(args.num):
            composites(i)
    elif args.type == "poly":
        for i in range(args.num):
            poly(i)
    else:
        print("Unknown dataset type '" + args.type + "'... Valid datasets are\n\teven-odd\n\tpowers\n\tcomposites\n\tstraightline")

if __name__ == "__main__":
    main()
