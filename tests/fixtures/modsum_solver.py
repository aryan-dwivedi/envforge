import re


def solve(prompt):
    numbers = [int(x) for x in re.findall(r"[0-9]+", prompt)]
    *terms, modulus = numbers
    return str(sum(terms) % modulus)
