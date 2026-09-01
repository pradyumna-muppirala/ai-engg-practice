---
name: python-file-to-skill-generator
description: Analyze a Python file, identify all callable functions and methods, infer the business purpose of the code, and generate an executable Claude skill in a new folder named after the Python file. Runs in a forked context and continuously loops until the user chooses to quit.
context: fork
---

# Python File → Claude Skill Generator

## Purpose

This skill converts a Python source file into a Claude skill.

The generated skill should:

- Be semantically intelligent.
- Analyze the implementation logic, not just signatures.
- Discover all callable functions and methods.
- Infer the purpose and capabilities of the code.
- Create a single skill exposing all discovered callables.
- Generate a complete `SKILL.md`.
- Generate instructions that allow the resulting skill to invoke and utilize the original Python file.
- Create a new folder named after the Python file and place the generated `SKILL.md` inside it.

The analysis scope is limited to the selected Python file only.

---

# Menu

Display the following menu:

```
Select an option:

1. Process Python file and create a new Claude skill
2. Quit
```

Wait for user input.

---

# Option 2

If the user selects:

```
2
```

Respond:

```
Exiting Python File → Claude Skill Generator.
```

Terminate the skill.

---

# Option 1

If the user selects:

```
1
```

Request:

```
Enter the path to the Python file:
```

Wait for input.

---

# Python Analysis Procedure

Read the specified Python file.

Analyze:

- All functions.
- All classes.
- All methods.
- Public callables.
- Private callables.
- Helper functions.
- Internal functions.

Do not exclude any callable discovered in the file.

For every callable:

1. Determine its purpose.
2. Determine its inputs.
3. Determine its outputs.
4. Determine side effects.
5. Infer how it should be used.
6. Infer its relationship to other callables.

Perform semantic analysis of the actual code implementation.

Do not rely solely on:

- Function names.
- Type hints.
- Docstrings.

Use actual implementation behavior when generating descriptions.

---

# Skill Generation Rules

Generate exactly one skill for the Python file.

Infer automatically:

- Skill name.
- Skill description.
- Skill capabilities.
- Usage instructions.
- Invocation patterns.

Create a folder:

```
<python-file-name>/
```

Example:

```
utilities.py
```

creates

```
utilities/
    SKILL.md
```

Only create:

```
SKILL.md
```

No additional files should be generated.

---

# Generated Skill Requirements

The generated skill must:

1. Describe the purpose of the Python file.
2. Describe all discovered callables.
3. Explain when each callable should be used.
4. Explain expected inputs.
5. Explain outputs.
6. Include execution guidance.
7. Reference the original Python file.
8. Enable Claude to invoke the underlying Python functionality.
9. Be suitable for direct Claude skill registration.

---

# Completion

After generation:

Display:

```
Skill generated successfully.

Output Folder:
<generated-folder>

Generated File:
<generated-folder>/SKILL.md
```

---

# Loop Reset Behavior

After successful generation:

1. Preserve the generated files on disk.
2. Discard the prior working conversation.
3. Treat all previous analysis, file contents, generated output, and user interactions as no longer active.
4. Return to the initial menu.
5. Behave as though only this skill definition exists in context.

Re-display:

```
Select an option:

1. Process Python file and create a new Claude skill
2. Quit
```

Repeat indefinitely until the user selects Option 2.