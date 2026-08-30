# String Operations

## Description

A string operations utility library for performing common string manipulations on text input. Provides functions for reversing, converting case, counting vowels, checking palindromes, and concatenating strings.

## Capabilities

This skill exposes six callable functions for string manipulation. Each function accepts string input(s) and returns either a transformed string, a count, or a boolean result.

## Reference File

- **Original Python File:** `StringOps.py`
- **Location:** `E:/ai-engg-practice/ai-engg-practice/utils/StringOps.py`

## Usage Instructions

To invoke any of the functions below, reference the original Python file path and import the functions:

```python
import sys
sys.path.append("E:/ai-engg-practice/ai-engg-practice")
from utils.StringOps import reverse_string, to_uppercase, to_lowercase, count_vowels, is_palindrome, concat_strings
```

## Functions

### 1. reverse_string

**Purpose:** Returns the reverse of the input string using Python slice notation.

**When to use:** When you need to reverse the characters of a string (e.g., for palindrome checks, encoding, or display purposes).

**Input:**
- `s` (str): The input string to reverse.

**Output:**
- Returns a string that is the reverse of the input.

**Implementation:**
```python
reverse_string(s) = s[::-1]
```

---

### 2. to_uppercase

**Purpose:** Converts all characters in the input string to uppercase.

**When to use:** When normalizing text for comparison, display, or storage where uppercase representation is required.

**Input:**
- `s` (str): The input string.

**Output:**
- Returns a string with all alphabetic characters converted to uppercase.

**Implementation:**
```python
to_uppercase(s) = s.upper()
```

---

### 3. to_lowercase

**Purpose:** Converts all characters in the input string to lowercase.

**When to use:** When normalizing text for case-insensitive comparison, pattern matching, or storage where lowercase representation is required.

**Input:**
- `s` (str): The input string.

**Output:**
- Returns a string with all alphabetic characters converted to lowercase.

**Implementation:**
```python
to_lowercase(s) = s.lower()
```

---

### 4. count_vowels

**Purpose:** Counts the number of vowel characters (a, e, i, o, u) in the input string, considering both uppercase and lowercase.

**When to use:** When analyzing text for vowel frequency, educational purposes, or linguistic analysis.

**Input:**
- `s` (str): The input string to analyze.

**Output:**
- Returns an integer representing the total count of vowel characters.

**Implementation:**
```python
count_vowels(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)
```

**Side effects:** None.

---

### 5. is_palindrome

**Purpose:** Checks whether the input string is a palindrome, ignoring non-alphanumeric characters and case sensitivity.

**When to use:** When detecting palindromes in text, which is useful for word games, linguistic validation, or data quality checks.

**Input:**
- `s` (str): The input string to check.

**Output:**
- Returns `True` if the cleaned string reads the same forward and backward, `False` otherwise.

**Implementation:**
```python
is_palindrome(s):
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]
```

**Side effects:** None. The function strips non-alphanumeric characters and converts to lowercase internally before comparison.

---

### 6. concat_strings

**Purpose:** Concatenates (joins) two strings end-to-end using the `+` operator.

**When to use:** When combining two pieces of text into a single string.

**Input:**
- `s1` (str): The first string.
- `s2` (str): The second string.

**Output:**
- Returns a single string that is the concatenation of `s1` followed by `s2`.

**Implementation:**
```python
concat_strings(s1, s2) = s1 + s2
```

**Side effects:** None.

---

## Execution Guidance

1. Ensure the Python interpreter has access to the file path: `E:/ai-engg-practice/ai-engg-practice/utils/StringOps.py`
2. Import the desired functions directly from the module.
3. Each function is stateless and can be called independently.
4. No external dependencies or third-party libraries are required beyond standard Python.
5. The module also contains example/demo code at the module level (lines 25-60) that demonstrates string slicing, f-strings, `split()`, `join()`, `replace()`, and `strip()` methods. These execute upon import but do not affect the callable functions.
