import json
import os
from datetime import datetime


TASKS_FILE = "tasks.json"

def load_tasks():
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r") as f:
            return json.load(f)
    return []

def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)

def add_task(tasks, title, priority="medium"):
    task = {
        "id": len(tasks) + 1,
        "title": title,
        "priority": priority,
        "status": "pending",
        "created": datetime.now().isoformat()
    }
    tasks.append(task)
    save_tasks(tasks)
    print(f"✓ Task added: {title} (Priority: {priority})")

def list_tasks(tasks):
    if not tasks:
        print("No tasks found.")
        return
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_tasks = sorted(tasks, key=lambda x: priority_order.get(x["priority"], 3))
    for task in sorted_tasks:
        status_icon = "✓" if task["status"] == "completed" else "○"
        print(f"{status_icon} [{task['id']}] {task['title']} ({task['status']}) - Priority: {task['priority']}")

def update_status(tasks, task_id, status):
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = status
            save_tasks(tasks)
            print(f"✓ Task {task_id} marked as {status}")
            return
    print("Task not found.")

def delete_task(tasks, task_id):
    for i, task in enumerate(tasks):
        if task["id"] == task_id:
            tasks.pop(i)
            save_tasks(tasks)
            print(f"✓ Task {task_id} deleted")
            return
    print("Task not found.")

def update_priority(tasks, task_id, priority):
    if priority not in ["high", "medium", "low"]:
        print("Invalid priority. Use 'high', 'medium', or 'low'.")
        return
    for task in tasks:
        if task["id"] == task_id:
            task["priority"] = priority
            save_tasks(tasks)
            print(f"✓ Task {task_id} priority updated to {priority}")
            return
    print("Task not found.")

def main():
    tasks = load_tasks()
    while True:
        print("\n--- Task Manager ---")
        print("1. Add task")
        print("2. List tasks")
        print("3. Mark complete")
        print("4. Delete task")
        print("5. Update priority")
        print("6. Exit")
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            title = input("Enter task title: ").strip()
            priority = input("Enter priority (high/medium/low) [default: medium]: ").strip().lower() or "medium"
            if title:
                add_task(tasks, title, priority)
        elif choice == "2":
            list_tasks(tasks)
        elif choice == "3":
            list_tasks(tasks)
            task_id = int(input("Enter task ID: "))
            update_status(tasks, task_id, "completed")
        elif choice == "4":
            list_tasks(tasks)
            task_id = int(input("Enter task ID: "))
            delete_task(tasks, task_id)
        elif choice == "5":
            list_tasks(tasks)
            task_id = int(input("Enter task ID: "))
            priority = input("Enter new priority (high/medium/low): ").strip().lower()
            update_priority(tasks, task_id, priority)
        elif choice == "6":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()