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

first="Hello"
second="World"

result = first + " " + second
print(result)  # Output: Hello World

def concat_strings(s1, s2):
    """Concatenates two strings."""
    return s1 + s2

text ="Python Programming"

print(text[0:6])    # Output: Python
print(text[-11:])   # Output: Programming

name = "Alice"
age = 25
info = f"My name is {name} and I am {age} years old."
print(info)  # Output: My name is Alice and I am 25 years old.

sentence = "The quick brown fox jumps over the lazy dog"
words = sentence.split()
print(words)  # Output: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'] 

joined_sentence = ' '.join(words)
print(joined_sentence)  # Output: The quick brown fox jumps over the lazy dog   

joined_sentence = "|".join(words)
print(joined_sentence)  # Output: The|quick|brown|fox|jumps|over|the|lazy|dog

sentence = sentence.replace("fox", "wolf")
print(sentence)  # Output: The quick brown fox jumps over the lazy dog

original = "  Hello, World!  "
trimmed = original.strip()  
print(f"'{trimmed}'")  # Output: 'Hello, World!'
