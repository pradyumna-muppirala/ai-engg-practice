import numpy as np

arr =np.array([1,2,3,4])
print(arr)

zeroes = np.zeros((3,4))
print(zeroes)

ones = np.ones((2,5))
print(ones) 

identity = np.eye(4)
print(identity)

range_arr = np.arange(10, 50, 5)
print(range_arr)

linspace_arr = np.linspace(0, 10, 5)
print(linspace_arr)

arr=np.array([1,2,3,4,5,6,7,8,9])
reshaped_arr = arr.reshape(3,3)
print(reshaped_arr)

arr = np.array([1,2,3,4,5])
expanded_arr = arr[:, np.newaxis]
print(expanded_arr)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
added_arr = np.add(arr1, arr2)
print(added_arr)
multiplied_arr = np.multiply(arr1, arr2)
print(multiplied_arr)

divided_arr = np.divide(arr2, arr1)
print(divided_arr)

sqrt_arr = np.sqrt(arr1)
print(sqrt_arr)
power_arr = np.power(arr1, 2)
print(power_arr)
sum_arr = np.sum(arr1)
print(sum_arr)
mean_arr = np.mean(arr1)
print(mean_arr)
max_arr = np.max(arr1)
print(max_arr)
min_arr = np.min(arr1)
print(min_arr)

print(arr)
print(arr[2])  # Accessing element at index 2
print(arr[-1])  # Accessing last element
print(arr[1:5])  # Slicing from index 1 to 4
print(arr[:3])  # Slicing first three elements

print(reshaped_arr)

reshared_arr1 = np.reshape(reshaped_arr, (9,1))
print(reshared_arr1)

a = np.arange(1,6)
b = np.arange(6,11)

stacked_arr = np.vstack((a, b))
print(stacked_arr)
print(a + b)
print(b - a)
print(a * b)
print(b / a)
hstacked_arr = np.hstack((a, b))
print(hstacked_arr)

arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
added_arr = np.add(arr1, arr2)
print(added_arr)
multiplied_arr = np.multiply(arr1, arr2)
print(multiplied_arr)
divided_arr = np.divide(arr1, arr2)
print(divided_arr)
T_arr1 = arr1.T
print(T_arr1)
dot_product = np.dot(arr1, arr2)
print(dot_product)
print("-- Additional Exercises --")
arr1 = np.arange(1, 17).reshape(4, 4)
print(arr1)
arr2 = arr1.T
print(arr2)
sum_arr = np.sum(arr1)
print(sum_arr)
max_arr = np.max(arr1)
print(max_arr)
normalize_arr = (arr1 - np.min(arr1)) / (np.max(arr1) - np.min(arr1))
print(normalize_arr)
print(arr1)
random_arr = np.random.rand(4, 4)
print(random_arr)
print(random_arr.min())
print(random_arr.max())
print(random_arr.mean())
print(random_arr.std())

#broadcasting of arrays in numpy - dimensiions are aligns from the right 
arr = np.array([1,2,3])
print(arr + 10)

arr1 = np.array([[1 ,2 , 3], [4 , 5, 6 ]])
arr2 = np.array([1 ,0 , 1])
print (arr1 + arr2)

#Aggregation functions
arr1 = np.array([[1 , 2, 3], [4, 5, 6]])
print("Sum => ", np.sum(arr1))
print("Mean =>", np.mean(arr1))
print("Max =>", np.max(arr1))
print("Std =>", np.std(arr1))
print("min => ", np.min(arr1))
print("sum of the rowas =>", np.sum(arr1, axis=1))
print("sum along columns => ", np.sum(arr1, axis=0))

#Boolean indexing and filtering
arr1 = np.array([1 , 2, 3, 4, 5, 6])
print(arr1[arr1 % 2 ==0])

arr1[arr1 % 3 == 0] = -1
print(arr1)

np.random.seed(20082074)
rand1 = np.random.rand(3,3)
print(rand1)

randint1 = np.random.randint(0, 1000000, size=(3,3))
print(randint1)

arr1 = np.arange(1, 10)
arr2 = np.reshape(arr1, (3,3))
print(arr2)
arr3 = np.array([1,2 ,3 ]).reshape((1,3))
print(arr3)
result_add = arr2 + arr3
print(result_add)

result_mult = arr2 * arr3
print(result_mult)

result_mult *= 2
print(result_mult)

dataset = np.random.randint(1, 51, size=(5,5))
print(dataset)
dataset[dataset > 25] = 0
print(dataset)

print("sum => ", np.sum(dataset))
print("mean => ", np.mean(dataset))
print("std => ", np.std(dataset))
print("min => ", np.min(dataset))
print("max => ", np.max(dataset))

dataset3D = np.random.randint(1,28, size=(3,3,3))
print(dataset3D)
print("sum => ", np.sum(dataset3D, axis=0))
print("sum => ", np.sum(dataset3D, axis=1))
print("sum => ", np.sum(dataset3D, axis=-1))

randfloatarr = np.random.rand(3,3)
print("Random float array => \n", randfloatarr)
min = np.min(randfloatarr)
max = np.max(randfloatarr)
varianceVal = max - min
randfloatarr2 = ((randfloatarr-min)/varianceVal)
print("Normalized rand float array => \n", randfloatarr2)

# Create a binary mask for values greater than a threshold
threshold = 0.5
binary_mask = randfloatarr2 > threshold
print("Binary mask => \n", binary_mask)





