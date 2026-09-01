# math_operations

A collection of basic mathematical operations for Claude Code.

## Operations Covered

| Operation | Description | Python Signature |
|-----------|-------------|------------------|
| `add` | Add two numbers | `add(num1, num2)` |
| `subtract` | Subtract second number from first | `subtract(num1, num2)` |
| `multiply` | Multiply two numbers | `multiply(num1, num2)` |
| `divide` | Divide first number by second; handles division by zero | `divide(num1, num2)` |
| `factorial` | Compute factorial of a non-negative integer | `factorial(n)` |
| `power` | Raise base to an exponent | `power(base, exponent)` |
| `sqrt` | Compute square root; handles negative inputs | `sqrt(n)` |
| `modulus` | Compute remainder of division | `modulus(num1, num2)` |
| `floor_divide` | Floor division of first number by second; handles division by zero | `floor_divide(num1, num2)` |
| `is_prime` | Check if a number is prime | `is_prime(n)` |
| `gcd` | Greatest common divisor of two numbers | `gcd(a, b)` |
| `lcm` | Least common multiple of two numbers | `lcm(a, b)` |
| `absolute` | Absolute value of a number | `absolute(n)` |
| `logarithm` | Logarithm of a number with specified base | `logarithm(n, base=10)` |

## Usage Patterns

### Basic Computation

You can ask Claude to perform any of these operations directly:

```
What is add(5, 3)?
Compute factorial(7).
What's the GCD of 48 and 18?
```

### Chained Operations

Claude can chain operations together naturally:

```
Compute the square root of 144, then add 5 to the result.
```

### Edge Case Handling

The operations include explicit error handling for common edge cases:

- **Division by zero**: `divide()` and `floor_divide()` return error strings when the divisor is 0
- **Negative square root**: `sqrt()` returns an error for negative inputs
- **Logarithm of non-positive numbers**: `logarithm()` returns an error for n <= 0
- **Factorial of negative/low values**: `factorial()` handles 0 and 1 as base cases returning 1

### Prime and Number Theory

Use `is_prime()`, `gcd()`, and `lcm()` for number theory tasks:

```
Is 97 prime?
What's the greatest common divisor of 56 and 98?
What's the least common multiple of 6 and 8?
```

## Integration Notes

These operations map directly to the functions defined in `math_operations.py`. When working with Claude Code, you can:

1. Reference the function names directly in prompts
2. Ask Claude to implement similar operations in other languages
3. Use these as building blocks for more complex mathematical reasoning
4. Combine with other Claude capabilities (search, analysis, code generation)