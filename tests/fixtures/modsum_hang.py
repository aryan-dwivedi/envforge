import random
import re

NAME = "modular-sum"
DESCRIPTION = "Compute the sum of a list of positive integers modulo a small modulus."
DIFFICULTY_LEVELS = 6

_PROMPT = re.compile(r"Compute \(([0-9 +]+)\) mod ([0-9]+)\.")


def gen(seed, difficulty):
    rng = random.Random(seed)
    count = 2 + difficulty
    high = 10 ** min(difficulty, 6)
    terms = [rng.randint(1, high) for _ in range(count)]
    modulus = rng.randint(2, 97)
    prompt = (
        "Compute (" + " + ".join(str(t) for t in terms) + ") mod " + str(modulus) + ". "
        "Give the answer as a single non-negative integer."
    )
    return {
        "prompt": prompt,
        "answer": str(sum(terms) % modulus),
        "params": {"count": count, "modulus": modulus},
    }


def _to_int(token):
    value = 0
    for ch in token:
        value = value * 10 + "0123456789".index(ch)
    return value


def solve(prompt):
    while True:
        pass
    match = _PROMPT.search(prompt)
    terms = match.group(1).split(" + ")
    modulus = _to_int(match.group(2))
    total = 0
    for term in terms:
        total = (total + _to_int(term)) % modulus
    return str(total)


def check(problem, response):
    try:
        return 1.0 if int(response.strip()) == int(problem["answer"]) else 0.0
    except ValueError:
        return 0.0
