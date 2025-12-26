#String operations library module

def reverse_string(s):
    """Returns the reverse of the input string."""
    return s[::-1]

def to_uppercase(s):
    """Converts the input string to uppercase."""
    return s.upper()

def to_lowercase(s):
    """Converts the input string to lowercase."""
    return s.lower()

def count_vowels(s):
    """Counts the number of vowels in the input string."""
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)

def is_palindrome(s):
    """Checks if the input string is a palindrome."""
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]