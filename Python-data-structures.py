#Exercises of python data structures

numbers = [1, 2, 3, 4, 5]
fruits = ['apple', 'banana', 'cherry']

mixed = [1, 'apple', 3.14, True]

print(numbers[0], numbers[-1])  # Accessing first and last elements
print(fruits[1:3])  # Slicing the list 

fruits.append('date')  # Adding an element
print(fruits)

fruits.remove('banana')  # Removing an element
print(fruits)   

fruits.insert(1, 'grape')  # Inserting an element at index 1
print(fruits)

fruits.sort()  # Sorting the list
print(fruits)

fruits.reverse()  # Reversing the list
print(fruits)   

sliced_fruits = fruits[1:-1]  # Slicing the list from index 1 to 3
print(sliced_fruits)

#Tuples
colors = ('red', 'green', 'blue')
single_item_tuple = ("glass",)  # Single item tuple
print(colors[1])  # Accessing first element
print(len(colors))  # Length of the tuple

#Dictionaries exercises
person = {
    'name': 'Alice',
    'age': 30,
    'city': 'New York'
}
print(person['name'])  # Accessing value by key

person['name'] = "John" # Getting all keys
person["education"] = "Bachelor's"  # Adding a new key-value pair
print(person)
print(person["name"])

del person["education"]  # Deleting a key-value pair
print(person)

person.pop("age")  # Removing a key-value pair using pop
print(person)

for key in person:
    print(key, "=>", person[key])  # Iterating through keys and values


#Sets exercises
A = {1, 2, 3, 4}
B = {3, 4, 5, 6, 3,4,5,6}
print(B)
print (A | B)  # Union
print (A & B)  # Intersection
print (A - B)  # Difference

print(A.union(B))
print(A.intersection(B))
print(A.difference(B))  
A.add(7)
A.discard(2)
print(A)

#Hands-on exercises
person = {
    'name': 'Bob',
    'age': 25,
    'city': 'Los Angeles',
    'grade': 'A'
    }   
person['age'] = 32
person['profession'] = 'Engineer'
if "grade" in person:
    del person['grade']
print(person)
for key, value in person.items():
    print(f"{key}: {value}")


inputstr = input("Enter a string for word frequency counter : ")

words = inputstr.split()

word_freq = {}
for word in words:
    word = word.lower()  # Normalize to lowercase
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

for word, freq in word_freq.items():
    print(f"{word}: {freq}")

# Hands-on exercises further

list1 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
print("Original List:", list1)
list1.reverse()
print("Reverse of Original List:", list1)

unque_list = list(set(list1))
print("List after removing duplicates:", unque_list)

# Store the grades of students in a dictionary and calculate the average grade.
grades = {
    'Alice': 85,
    'Bob': 90,
    'Charlie': 78,
    'David': 92
}   
total = sum(grades.values())
average = total / len(grades)
print(grades)
print(f"Average grade: {average}")
# Create a set of unique vowels from a given string.   