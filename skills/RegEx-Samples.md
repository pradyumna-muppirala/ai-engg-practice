# Regular Expression Samples

## Description

A regular expressions utility module that provides functions for cleaning text by removing special characters and redacting email addresses. Serves as an exercise file demonstrating regex pattern matching, substitution, and text manipulation.

## Capabilities

This skill exposes two callable functions for regex-based text processing. Each function accepts string input and returns a transformed string using Python's `re` module.

## Reference File

- **Original Python File:** `RegEx-Samples.py`
- **Location:** `E:/ai-engg-practice/ai-engg-practice/utils/RegEx-Samples.py`

## Usage Instructions

To invoke any of the functions below, reference the original Python file path and import the functions:

```python
import sys
sys.path.append("E:/ai-engg-practice/ai-engg-practice")
from utils.RegEx_Samples import clean_the_text, replace_emails
```

## Functions

### 1. clean_the_text

**Purpose:** Removes special characters (punctuation, symbols) from the input text, preserving only word characters (`\w`) and whitespace (`\s`).

**When to use:** When you need to sanitize text by stripping punctuation and symbols, such as preparing text for analysis, cleaning user input, or normalizing data.

**Input:**
- `input_text` (str): The input text to clean.

**Output:**
- Returns a string with all non-word, non-whitespace characters removed.

**Implementation:**
```python
clean_the_text(input_text):
    return re.sub(r'[^\w\s]', '', input_text)
```

**Example:**
```python
clean_the_text(" Hello, World! Welcome  to  RegEx 101.")
# Output: " Hello World Welcome  to  RegEx 101"
```

**Side effects:** None.

---

### 2. replace_emails

**Purpose:** Replaces email addresses found in the text with a placeholder string, preventing email leakage in logs, messages, or shared content.

**When to use:** When redacting or masking email addresses in text before logging, displaying, or sharing content.

**Input:**
- `text` (str): The input text containing email addresses.
- `replacement` (str, optional): The placeholder to substitute for each email. Defaults to `"[EMAIL REDACTED]"`.

**Output:**
- Returns a new string with all email addresses replaced by the placeholder.

**Implementation:**
```python
replace_emails(text, replacement="[EMAIL REDACTED]"):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.sub(email_pattern, replacement, text)
```

**Example:**
```python
replace_emails("Please contact us at abc@bcd.cin or xyz@yza.za")
# Output: "Please contact us at [EMAIL REDACTED] or [EMAIL REDACTED]"
```

**Side effects:** None. The original text is not modified; a new string is returned.

---

## Execution Guidance

1. Ensure the Python interpreter has access to the file path: `E:/ai-engg-practice/ai-engg-practice/utils/RegEx-Samples.py`
2. Import the desired functions directly from the module.
3. Each function is stateless and can be called independently.
4. No external dependencies or third-party libraries are required beyond the standard `re` module.
5. **Note:** The module also imports `StringOps` and uses `StringOps.count_vowels` in its demo code. When imported, the module's top-level code executes, which may print demo output. This does not affect the callable functions.

## Related Skills

- **StringOps** — Provides string manipulation functions (`reverse_string`, `to_uppercase`, `to_lowercase`, `count_vowels`, `is_palindrome`, `concat_strings`). The `RegEx-Samples` module demonstrates integration with `StringOps.count_vowels` for vowel counting on regex-processed text.