"""
Unit tests for math_operations.py
Run from repository root with: pytest tests/test_math_operations.py -v
"""

import os
import sys

# Add the math directory to the path so we can import math_operations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'math'))

import pytest
# Import math for math.e in test
import math
import math_operations as mo


class TestAdd:
    """Tests for the add function."""

    def test_positive_numbers(self):
        assert mo.add(2, 3) == 5

    def test_negative_numbers(self):
        assert mo.add(-2, -3) == -5

    def test_mixed_numbers(self):
        assert mo.add(-2, 3) == 1

    def test_zero(self):
        assert mo.add(5, 0) == 5
        assert mo.add(0, 5) == 5


class TestSubtract:
    """Tests for the subtract function."""

    def test_positive_numbers(self):
        assert mo.subtract(5, 3) == 2

    def test_negative_numbers(self):
        assert mo.subtract(-5, -3) == -2

    def test_mixed_numbers(self):
        assert mo.subtract(3, 5) == -2

    def test_zero(self):
        assert mo.subtract(5, 0) == 5
        assert mo.subtract(0, 5) == -5


class TestMultiply:
    """Tests for the multiply function."""

    def test_positive_numbers(self):
        assert mo.multiply(2, 3) == 6

    def test_negative_numbers(self):
        assert mo.multiply(-2, -3) == 6
        assert mo.multiply(-2, 3) == -6

    def test_zero(self):
        assert mo.multiply(5, 0) == 0
        assert mo.multiply(0, 5) == 0


class TestDivide:
    """Tests for the divide function."""

    def test_positive_numbers(self):
        assert mo.divide(6, 2) == 3.0

    def test_negative_numbers(self):
        assert mo.divide(-6, -2) == 3.0

    def test_mixed_numbers(self):
        assert mo.divide(-6, 2) == -3.0

    def test_fractional_result(self):
        assert mo.divide(5, 2) == 2.5

    def test_division_by_zero_returns_error_string(self):
        """divide returns error string for division by zero."""
        result = mo.divide(5, 0)
        assert result == "Error! Division by zero."
        assert isinstance(result, str)

    def test_division_by_zero_with_negative(self):
        """divide returns error string for negative divisor zero."""
        result = mo.divide(-5, 0)
        assert result == "Error! Division by zero."


class TestFactorial:
    """Tests for the factorial function."""

    def test_factorial_zero(self):
        assert mo.factorial(0) == 1

    def test_factorial_one(self):
        assert mo.factorial(1) == 1

    def test_factorial_positive(self):
        assert mo.factorial(5) == 120
        assert mo.factorial(10) == 3628800

    def test_factorial_negative_raises_recursion_error(self):
        """factorial with negative input causes RecursionError."""
        with pytest.raises(RecursionError):
            mo.factorial(-1)


class TestPower:
    """Tests for the power function."""

    def test_positive_exponent(self):
        assert mo.power(2, 3) == 8

    def test_zero_exponent(self):
        assert mo.power(5, 0) == 1

    def test_negative_exponent(self):
        assert mo.power(2, -2) == 0.25

    def test_zero_base(self):
        assert mo.power(0, 5) == 0

    def test_fractional_exponent(self):
        assert mo.power(4, 0.5) == 2.0


class TestSqrt:
    """Tests for the sqrt function."""

    def test_sqrt_of_perfect_square(self):
        assert mo.sqrt(4) == 2.0
        assert mo.sqrt(9) == 3.0

    def test_sqrt_of_non_perfect_square(self):
        assert mo.sqrt(2) == pytest.approx(1.41421356237, rel=1e-9)

    def test_sqrt_of_zero(self):
        assert mo.sqrt(0) == 0.0

    def test_sqrt_of_negative_returns_error_string(self):
        """sqrt returns error string for negative input."""
        result = mo.sqrt(-1)
        assert result == "Error! Cannot compute square root of negative number."
        assert isinstance(result, str)


class TestModulus:
    """Tests for the modulus function."""

    def test_positive_numbers(self):
        assert mo.modulus(10, 3) == 1

    def test_negative_dividend(self):
        # Python's modulo returns result with sign of divisor: -10 % 3 == 2
        assert mo.modulus(-10, 3) == 2

    def test_division_by_zero_raises_error(self):
        """modulus raises ZeroDivisionError for zero divisor."""
        with pytest.raises(ZeroDivisionError):
            mo.modulus(5, 0)


class TestFloorDivide:
    """Tests for the floor_divide function."""

    def test_positive_numbers(self):
        assert mo.floor_divide(7, 2) == 3

    def test_division_by_zero_returns_error_string(self):
        """floor_divide returns error string for division by zero."""
        result = mo.floor_divide(5, 0)
        assert result == "Error! Division by zero."
        assert isinstance(result, str)


class TestIsPrime:
    """Tests for the is_prime function."""

    def test_prime_numbers(self):
        assert mo.is_prime(2) == True
        assert mo.is_prime(3) == True
        assert mo.is_prime(5) == True
        assert mo.is_prime(7) == True
        assert mo.is_prime(11) == True
        assert mo.is_prime(13) == True

    def test_non_prime_numbers(self):
        assert mo.is_prime(4) == False
        assert mo.is_prime(6) == False
        assert mo.is_prime(8) == False
        assert mo.is_prime(9) == False
        assert mo.is_prime(10) == False

    def test_edge_cases(self):
        assert mo.is_prime(0) == False
        assert mo.is_prime(1) == False
        assert mo.is_prime(-5) == False


class TestGcd:
    """Tests for the gcd function."""

    def test_same_numbers(self):
        assert mo.gcd(5, 5) == 5

    def test_different_numbers(self):
        assert mo.gcd(12, 8) == 4
        assert mo.gcd(15, 10) == 5

    def test_one_is_multiple(self):
        assert mo.gcd(5, 25) == 5

    def test_coprime_numbers(self):
        assert mo.gcd(7, 11) == 1


class TestLcm:
    """Tests for the lcm function."""

    def test_same_numbers(self):
        assert mo.lcm(5, 5) == 5

    def test_different_numbers(self):
        assert mo.lcm(4, 6) == 12
        assert mo.lcm(3, 5) == 15

    def test_one_is_multiple(self):
        assert mo.lcm(5, 25) == 25

    def test_with_negative_numbers(self):
        """lcm uses absolute values."""
        assert mo.lcm(-4, 6) == 12
        assert mo.lcm(4, -6) == 12
        assert mo.lcm(-4, -6) == 12


class TestAbsolute:
    """Tests for the absolute function."""

    def test_positive_number(self):
        assert mo.absolute(5) == 5

    def test_negative_number(self):
        assert mo.absolute(-5) == 5

    def test_zero(self):
        assert mo.absolute(0) == 0


class TestLogarithm:
    """Tests for the logarithm function."""

    def test_logarithm_base_10(self):
        assert mo.logarithm(100, 10) == 2.0
        assert mo.logarithm(10, 10) == 1.0

    def test_logarithm_base_e(self):
        # natural logarithm: log base e of e == 1
        assert mo.logarithm(math.e, math.e) == pytest.approx(1.0, rel=1e-9)

    def test_logarithm_base_2(self):
        assert mo.logarithm(8, 2) == 3.0

    def test_logarithm_default_base(self):
        assert mo.logarithm(1000) == pytest.approx(3.0, rel=1e-6)

    def test_logarithm_one_returns_error(self):
        """logarithm returns error string for n=1."""
        # Actually log(1, base) = 0, which is valid
        assert mo.logarithm(1, 10) == 0.0

    def test_logarithm_negative_n_returns_error_string(self):
        """logarithm returns error string for negative input."""
        result = mo.logarithm(-10, 10)
        assert result == "Error! Logarithm undefined for non-positive numbers."
        assert isinstance(result, str)

    def test_logarithm_zero_returns_error_string(self):
        """logarithm returns error string for zero input."""
        result = mo.logarithm(0, 10)
        assert result == "Error! Logarithm undefined for non-positive numbers."
        assert isinstance(result, str)

    def test_logarithm_invalid_base_zero(self):
        """logarithm with base=0 raises ValueError."""
        with pytest.raises(ValueError):
            mo.logarithm(100, 0)

    def test_logarithm_invalid_base_one(self):
        """logarithm with base=1 raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            mo.logarithm(100, 1)

    def test_logarithm_negative_base(self):
        """logarithm with negative base raises ValueError."""
        with pytest.raises(ValueError):
            mo.logarithm(100, -2)


