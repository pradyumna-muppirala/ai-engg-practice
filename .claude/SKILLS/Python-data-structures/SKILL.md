# Python-data-structures

Exercises covering fundamental Python data structures including lists, tuples, dictionaries, and sets for Claude Code.

## Operations Covered

| Operation | Description | Python Signature |
|-----------|-------------|------------------|
| `list_access` | Access elements by index | `list_access(numbers, index)` |
| `list_slicing` | Slice lists | `list_slicing(fruits, start, end)` |
| `list_modification` | Modify lists (append, remove, insert) | `list_modification(fruits, operation, value)` |
| `list_sorting` | Sort and reverse lists | `list_sorting(fruits)` |
| `tuple_access` | Access tuple elements | `tuple_access(colors, index)` |
| `dictionary_operations` | Access, modify, delete dictionary keys | `dictionary_operations(person, key, value)` |
| `set_operations` | Union, intersection, difference, add, discard | `set_operations(A, B, operation)` |
| `word_frequency_counter` | Count word frequencies from input string | `word_frequency_counter(input_str)` |
| `unique_list` | Remove duplicates from a list | `unique_list(list1)` |
| `grade_average` | Calculate average grade from dictionary | `grade_average(grades)` |

## Core Concepts Demonstrated

### Lists
- **Indexing and Slicing**: Access first/last elements, slice ranges
- **Modification**: append, remove, insert operations
- **Sorting and Reversing**: Built-in sort() and reverse() methods
- **Duplicate Removal**: Using set() to create unique lists

### Tuples
- **Creation and Access**: Create tuples, single-item tuples, element access
- **Length**: Using len() to get tuple length

### Dictionaries
- **Key-Value Access**: Access values by key, modify values
- **Adding Keys**: Add new key-value pairs
- **Deleting Keys**: Using del and pop() methods
- **Iteration**: Iterating through keys and values with items()

### Sets
- **Set Operations**: Union (|), Intersection (&), Difference (-)
- **Membership Testing**: Using `in` operator
- **Adding and Discarding**: add() and discard() methods
- **Word Frequency Counter**: Count occurrences of each word

## Usage Patterns

### List Operations

You can ask Claude to demonstrate list operations:

```
Show me list operations from Python-data-structures.py
How do I remove duplicates from a list in Python?
```

### Dictionary Operations

Claude can demonstrate dictionary manipulations:

```
How do I add and remove keys from a dictionary?
Show me how to iterate through a dictionary
```

### Set Operations

The file includes set operations examples:

```
What's the difference between union and intersection in Python sets?
Show me set operations from the exercises
```

### Word Frequency Analysis

The file includes a hands-on word frequency counter:

```
Count word frequencies from a sample text
```

## Integration Notes

These examples map directly to the code defined in `Python-data-structures.py`. When working with Claude Code, you can:

1. Reference the function names and code snippets directly in prompts
2. Ask Claude to explain data structure concepts demonstrated in the exercises
3. Use these as templates for learning Python data structures
4. Modify and extend the examples for practice
5. Run the examples to see them in action