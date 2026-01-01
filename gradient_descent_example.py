import sympy as sp
import numpy as np

# -----------------------------------------------------------------------------
# Section 8.1: Symbolic Definition Module
# -----------------------------------------------------------------------------
def generate_symbolic_functions():
    """
    Constructs the computational graph using SymPy.
    
    Returns:
        tuple: (func_grad_a, func_grad_b, func_loss, func_predict)
               These are optimized NumPy functions compiled from symbolic logic.
    """
    # 1. Define Symbols
    # We use generic symbols. 'x' and 'y' represent the input features.
    # 'z' represents the target variable. 'a' and 'b' are the coefficients.
    a, b, x, y, z = sp.symbols('a b x y z')
    
    # 2. Define the Hypothesis
    # The linear model: z_pred = a*x + b*y
    prediction = a * x + b * y
    
    # 3. Define the Element-wise Loss
    # We use Squared Error: SE = (z_observed - z_predicted)^2
    # Note: We do not sum over N here. We compute the derivative of the 
    # error for a single theoretical point. The averaging happens in NumPy.
    # This allows for efficient vectorization.
    squared_error = (z - prediction)**2
    
    # 4. Symbolic Differentiation
    # SymPy automatically applies the chain rule.
    # diff(error, a) -> 2 * (z - (ax+by)) * (-x)
    grad_a_sym = sp.diff(squared_error, a)
    grad_b_sym = sp.diff(squared_error, b)
    
    # Debug output to verify the symbolic math
    print(f" Hypothesis: {prediction}")
    print(f" Gradient w.r.t a: {grad_a_sym}")
    print(f" Gradient w.r.t b: {grad_b_sym}")
    
    # 5. Lambdification (Compilation)
    # convert the symbolic expressions into Python functions.
    # modules='numpy' ensures that the math operators use efficient array routines.
    # The resulting functions will accept (a, b, x, y, z) as arguments.
    
    print(" Compiling symbolic expressions to NumPy functions...")
    
    # Gradient functions
    func_grad_a = sp.lambdify((a, b, x, y, z), grad_a_sym, modules='numpy')
    func_grad_b = sp.lambdify((a, b, x, y, z), grad_b_sym, modules='numpy')
    
    # Prediction and Loss functions (for monitoring)
    func_predict = sp.lambdify((a, b, x, y), prediction, modules='numpy')
    func_loss = sp.lambdify((a, b, x, y, z), squared_error, modules='numpy')
    
    return func_grad_a, func_grad_b, func_loss, func_predict

# -----------------------------------------------------------------------------
# Section 8.2: Gradient Descent Engine
# -----------------------------------------------------------------------------
def gradient_descent_optimization(x_list, y_list, z_list, 
                                  iterations=1000, 
                                  learning_rate=0.1, 
                                  tolerance=1e-6):
    """
    Performs gradient descent to optimize z = a*x + b*y.
    
    Args:
        x_list, y_list: List of Lists (Input features)
        z_list: List (Target values)
        iterations: Max number of update steps
        learning_rate: Step size multiplier (alpha)
        
    Returns:
        dict: Final parameters and history
    """
    
    # 1. Data Type Conversion (Critical Performance Step)
    # Convert "list of lists" to contiguous memory blocks (ndarrays).
    # This enables SIMD vectorization in the subsequent calculations.
    X = np.array(x_list)
    Y = np.array(y_list)
    Z = np.array(z_list)
    
    # Shape Validation & Broadcasting Prep
    # If Z is 1D (flat list) and X is 2D, we must reshape Z to match X 
    # to avoid ambiguous broadcasting (e.g., subtracting row vector from column vector).
    if Z.ndim == 1 and X.ndim == 2:
        if Z.size == X.size:
            Z = Z.reshape(X.shape)
            print(f" Reshaped Z from flat list to {Z.shape} grid.")
        else:
            raise ValueError(f"Shape mismatch: X is {X.shape} but Z has {Z.size} elements.")
            
    # 2. Symbolic Compilation
    # Generate the high-performance functions
    grad_a_func, grad_b_func, loss_func, pred_func = generate_symbolic_functions()
    
    # 3. Initialization
    # Initialize coefficients randomly from a normal distribution.
    # Random seed is not set here to allow variability in production use, 
    # but set in testing for reproducibility.
    a_est = np.random.randn()
    b_est = np.random.randn()
    
    print(f"\n[Init] Starting Parameters: a={a_est:.4f}, b={b_est:.4f}")
    
    loss_history = []
    param_history = []
    
    # 4. The Optimization Loop
    for i in range(iterations):
        # A. Gradient Computation
        # We pass the full arrays X, Y, Z. The lambdified function returns
        # a matrix of element-wise gradients.
        grad_a_elements = grad_a_func(a_est, b_est, X, Y, Z)
        grad_b_elements = grad_b_func(a_est, b_est, X, Y, Z)
        
        # B. Aggregation (Batch Gradient Descent)
        # The true gradient of the Mean Squared Error is the MEAN of the element-wise gradients.
        # Note: If our symbolic loss was Sum of Squared Errors, we would use sum().
        # Since we want MSE, and our symbolic derivative is for a single point, 
        # the mean is appropriate.
        d_a = np.mean(grad_a_elements)
        d_b = np.mean(grad_b_elements)
        
        # C. Parameter Update
        a_est = a_est - learning_rate * d_a
        b_est = b_est - learning_rate * d_b
        
        # D. Monitoring
        if i % 100 == 0 or i == iterations - 1:
            # Compute current MSE for logging
            # (Calculation is vectorized)
            current_errors = loss_func(a_est, b_est, X, Y, Z)
            current_mse = np.mean(current_errors)
            loss_history.append(current_mse)
            param_history.append((a_est, b_est))
            
            print(f"Iter {i:04d} | Loss: {current_mse:.6f} | Grads: ({d_a:.4f}, {d_b:.4f}) | a: {a_est:.4f}, b: {b_est:.4f}")
            
            # Convergence Check (Optional early stopping)
            if current_mse < tolerance:
                print("Converged early.")
                break
                
    return {
        "final_a": a_est,
        "final_b": b_est,
        "loss_history": loss_history
    }

# -----------------------------------------------------------------------------
# Section 8.3: Testing Harness
# -----------------------------------------------------------------------------
def run_test_scenario():
    print("=== Gradient Descent Test Harness ===")
    
    # 1. Setup Ground Truth
    true_a = 2.5
    true_b = -1.5
    print(f"Target Ground Truth: a={true_a}, b={true_b}")
    
    # 2. Generate Synthetic Data
    # Dimensions: 10x10 grid (List of Lists structure)
    # rows, cols = 10, 10
    np.random.seed(42) # Ensure reproducibility for the report
    rows = 1
    cols = 3
    # Create random inputs between 0 and 2
    # x_numpy = np.random.rand(rows, cols) * 2
    # y_numpy = np.random.rand(rows, cols) * 2

    x_numpy = [1 , 2, 3]
    y_numpy = [2 , 3, 4]
    
    # Calculate Z = a*x + b*y + noise
    # noise_level = 0.05
    # noise = np.random.randn(rows, cols) * noise_level
    # z_numpy = (true_a * x_numpy) + (true_b * y_numpy) + noise
    
    z_numpy = [3 , 3.5, 4]
    # Convert to standard Python "list of lists" as per requirements
    x_input = x_numpy
    y_input = y_numpy
    z_input = z_numpy
    
    print(f"Input Data: {rows}x{cols} grid generated.")
    
    # 3. Execute Optimization
    results = gradient_descent_optimization(
        x_input, 
        y_input, 
        z_input, 
        iterations=1000, 
        learning_rate=0.1
    )
    
    # 4. Validate Results
    final_a = results["final_a"]
    final_b = results["final_b"]
    
    print("\n=== Final Results Validation ===")
    print(f"Estimated a: {final_a:.5f} (True: {true_a}) -> Error: {abs(final_a - true_a):.5f}")
    print(f"Estimated b: {final_b:.5f} (True: {true_b}) -> Error: {abs(final_b - true_b):.5f}")
    
    # Comparison Table
    print("\n| Parameter | True Value | Estimated | Abs Error |")
    print("|-----------|------------|-----------|-----------|")
    print(f"| a | {true_a:10.4f} | {final_a:9.4f} | {abs(final_a-true_a):9.4f} |")
    print(f"| b | {true_b:10.4f} | {final_b:9.4f} | {abs(final_b-true_b):9.4f} |")

if __name__ == "__main__":
    run_test_scenario()