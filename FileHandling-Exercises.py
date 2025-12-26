import os

try:
    with open("data.txt", "r+") as file:
        lines = file.readlines()
        file.seek(0)
        for line in lines:
            print(line)
            line2 = line.replace("Hello", "Hi")
            file.write(line2)
        file.truncate()
except FileNotFoundError:
    print("The file 'data.txt' does not exist.")
except IOError:
    print("An error occurred while handling the file.")
except PermissionError:
    print("You do not have permission to read/write the file.") 
except Exception as e:
    print(f"An unexpected error occurred: {e}")
# Ensure the file is properly closed after operations
finally:
    if 'file' in locals() and not file.closed:
        file.close()