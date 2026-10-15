# Task: write a procedural problem generator

Write a Python module that generates, solves and checks instances of this problem family.

## Problem family

Title: {title}

{description}

## Module contract

The module must define exactly these names:

```python
{contract}
```

## Allowed imports

Only these modules may be imported: {allowlist}

These names may not appear anywhere in the module: {banned}. Attribute access to dunder names (`__x__`) is not allowed.

## Requirements

- `solve` must derive the answer from the prompt text alone
- `check` must parse `response` tolerantly (strip whitespace; numeric answers compared as numbers; fractions as `p/q` in lowest terms) and return exactly 0.0 or 1.0
- prompt must state the exact answer format
- difficulty must monotonically increase problem size/search space
- respond with one ```python block

Respond with exactly one ```python fenced block containing the full module.
