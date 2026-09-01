# HelloWorld

A collection of basic programming examples and data structure operations for Claude Code.

## Operations Covered

| Operation | Description | Python Signature |
|-----------|-------------|------------------|
| `print_hello_world` | Prints "Hello, World!" | `print("Hello, World!")` |
| `quicksort` | Implements quicksort algorithm | `quicksort(arr)` |
| `elementwise_mul_2x2` | Returns element-wise product of two 2x2 matrices | `elementwise_mul_2x2(A, B)` |
| `matrix_mul_2x2` | Returns matrix product of two 2x2 matrices | `matrix_mul_2x2(A, B)` |
| `merge_sort` | Implements merge sort algorithm with type hints | `merge_sort(arr: list) -> list` |
| `merge` | Helper function to merge two sorted lists | `merge(left: list, right: list) -> list` |

## Usage Patterns

### Basic Output

You can ask Claude to demonstrate basic output:

```
Run HelloWorld.py to see the Hello World message
```

### Sorting Algorithms

Claude can explain and demonstrate sorting algorithms:

```
Explain how quicksort works with the example in HelloWorld.py
Show me how merge_sort sorts the list [64, 34, 25, 12, 22, 11, 90, 91, 33, 56, 67, 34]
```

### Matrix Operations

The file includes examples of 2x2 matrix operations:

```
What is the element-wise product of [[1,2],[3,4]] and [[5,6],[7,8]]?
What is the matrix product of the same matrices?
```

### Code Examples

The file contains runnable examples that demonstrate:

- Basic string output
- Quicksort implementation
- 2x2 matrix operations (element-wise and standard multiplication)
- Merge sort with type hints
- Helper functions for sorting

## Integration Notes

These examples map directly to the code defined in `HelloWorld.py`. When working with Claude Code, you can:

1. Reference the function names directly in prompts
2. Ask Claude to explain the algorithms shown
3. Use these as building blocks for learning programming concepts
4. Modify and extend the examples for practice