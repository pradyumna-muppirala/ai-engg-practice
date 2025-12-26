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

def count_words_in_file(filename):
    #count the words in the given input file by opening the file in read mode
    word_count = 0
    try:
        with open(filename, "r") as file:
            content = file.read()
            words = content.split()
            word_count = len(words)
            print(f"The file contains {word_count} words.")
    except FileNotFoundError:
        print("The file 'data.txt' does not exist.")
    except IOError:
        print("An error occurred while handling the file.")
    except PermissionError:
        print("You do not have permission to read the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'file' in locals() and not file.closed:
            file.close()
    return word_count

count_words_in_file("data.txt")

# function to write a list of items and read them back, print them on terminal
def write_and_read_items(filename, items):
    try:
        with open(filename, "w+") as file:
            for item in items:
                file.write(f"{item}\n")
            file.seek(0)
            print("Items in the file:")
            for line in file:
                print(line.strip())
    except IOError:
        print("An error occurred while handling the file.")
    except PermissionError:
        print("You do not have permission to read/write the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'file' in locals() and not file.closed:
            file.close()

items_list = ["Apple", "Banana", "Cherry", "Date"]
items_list.append("Elderberry")
write_and_read_items("items.txt", items_list)

# function to copy content from one file to another
def file_copy(source, destination):
    try:
        with open(source, "r") as src_file:
            content = src_file.read()
        with open(destination, "w") as dest_file:
            dest_file.write(content)
        print(f"Content copied from {source} to {destination}.")
    except FileNotFoundError:
        print(f"The file '{source}' does not exist.")
    except IOError:
        print("An error occurred while handling the file.")
    except PermissionError:
        print("You do not have permission to read/write the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'src_file' in locals() and not src_file.closed:
            src_file.close()
        if 'dest_file' in locals() and not dest_file.closed:
            dest_file.close()

file_copy("data.txt", "data_copy.txt")

# function to count the number of occurences of a specific word in a file
def count_word_occurrences(filename, target_word):
    count = 0
    try:
        with open(filename, "r") as file:
            for line in file:
                words = line.split()
                count += words.count(target_word)
        print(f"The word '{target_word}' occurs {count} times in the file.")
    except FileNotFoundError:
        print(f"The file '{filename}' does not exist.")
    except IOError:
        print("An error occurred while handling the file.") 
    except PermissionError:
        print("You do not have permission to read the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'file' in locals() and not file.closed:
             file.close()
    return count
    
count_word_occurrences("data.txt", "Hi")

#write program to append log messages with timestamps into a log file, by taking input from user 

from datetime import datetime
def append_log_message(logfile, message):
    try:
        with open(logfile, "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"[{timestamp}] {message}\n")
        print("Log message appended.")
    except IOError:
        print("An error occurred while handling the file.")
    except PermissionError:
        print("You do not have permission to write to the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'file' in locals() and not file.closed:
            file.close()    
    return

isexit = False

while not isexit:
    user_message = input("Enter a log message (or type 'exit' to quit): ")
    if user_message.lower() == 'exit':
        isexit = True
    else:
        append_log_message("app_log.txt", user_message)
