#Regular expressions exercises
import re
import StringOps

text = "Contact me at 123-456-7890 or at 987-654-3210."

digits=re.findall(r'\d', text)
print("Digits found:", digits)

updated_text = re.sub(r'\d', 'X', text)
print("Updated text:", updated_text)    

def clean_the_text(input_text):
    """Removes special characters from the input text."""
    return re.sub(r'[^\w\s]', '', input_text)

sample_text = " Hello, World! Welcome  to  RegEx 101."
cleaned_text = clean_the_text(sample_text)
print("Cleaned text:", cleaned_text)  # Output: Hello World Welcome to RegEx 101

str1 = "madam"
print(f"Is '{str1}' a palindrome? {str1 == str1[::-1]}")  # Output: True
str2 = "amanaplanacanalpanama"
print(f"Is '{str2}' a palindrome? {str2 == str2[::-1]}")  # Output: True

print("Vowels in {str2} : ", StringOps.count_vowels(str2))  # Output: 10

def replace_emails(text, replacement="[EMAIL REDACTED]"):
    """Replaces email addresses in the text with a placeholder."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.sub(email_pattern, replacement, text)

email_text = "Please contact us at abc@bcd.cin or xyz@yza.za or pqr@qrs.ru for more informatioon."

print(replace_emails(email_text))


