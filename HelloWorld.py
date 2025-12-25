print("Hello, World!")

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Example usage
numbers = [64, 34, 25, 12, 22, 11, 90, 91, 33, 56, 67, 34]
sorted_numbers = quicksort(numbers)
print(f"Sorted list: {sorted_numbers}")


def elementwise_mul_2x2(A, B):
    """Return the element-wise product of two 2x2 matrices A and B.

    A and B must be lists of two lists, each with two numeric elements.
    Example: A = [[a,b],[c,d]]
    """
    if (not isinstance(A, list) or not isinstance(B, list)
            or len(A) != 2 or len(B) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in A)
            or any(not isinstance(row, list) or len(row) != 2 for row in B)):
        raise ValueError("A and B must be 2x2 matrices (lists of two lists of two numbers).")

    return [
        [A[0][0] * B[0][0], A[0][1] * B[0][1]],
        [A[1][0] * B[1][0], A[1][1] * B[1][1]],
    ]


if __name__ == "__main__":
    # Example matrices
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = elementwise_mul_2x2(A, B)
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"Element-wise product C = {C}")

    def matrix_mul_2x2(A, B):
        """Return the matrix product of two 2x2 matrices A and B.
        
        A and B must be lists of two lists, each with two numeric elements.
        """
        if (not isinstance(A, list) or not isinstance(B, list)
                or len(A) != 2 or len(B) != 2
                or any(not isinstance(row, list) or len(row) != 2 for row in A)
                or any(not isinstance(row, list) or len(row) != 2 for row in B)):
            raise ValueError("A and B must be 2x2 matrices (lists of two lists of two numbers).")
        
        return [
            [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
        ]


    # Example matrices
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    D = matrix_mul_2x2(A, B)
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"Matrix product D = {D}")