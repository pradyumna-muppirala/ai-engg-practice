import os
import subprocess
from anthropic import Anthropic

# Initialize client pointing to OpenRouter
client = Anthropic(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Define local tool schema available to the agent
tools = [
    {
        "name": "run_bash",
        "description": "Execute a local shell command to test or verify code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."}
            },
            "required": ["command"]
        }
    }
]

def execute_bash(command: str) -> str:
    """Local tool execution engine."""
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return res.stdout if res.returncode == 0 else f"ERROR: {res.stderr}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

def run_agentic_loop(prompt: str, max_turns: int = 10):
    messages = [{"role": "user", "content": prompt}]
    
    for turn in range(max_turns):
        print(f"\n--- [Turn {turn + 1}] Invoking poolside/laguna-s-2.1:free ---")
        
        # 1. Reason & Plan Call
        response = client.messages.create(
            model="poolside/laguna-s-2.1:free",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )
        
        # 2. Check LOOP EXIT CONDITION: Stop reason is end_turn and no tool calls exist
        if response.stop_reason == "end_turn" and not any(block.type == "tool_use" for block in response.content):
            print("\n✅ Exit Condition Signaled: Task Completed.")
            final_text = "".join([block.text for block in response.content if block.type == "text"])
            print(f"Final Agent Response:\n{final_text}")
            break

        # 3. Handle Tool Executions if tool_use block is present
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        
        for block in response.content:
            if block.type == "tool_use":
                print(f"🔧 Executing Tool: {block.name} with input: {block.input}")
                if block.name == "run_bash":
                    output = execute_bash(block.input["command"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output
                    })
        
        # Feed local tool stdout/stderr back into the conversation context
        messages.append({"role": "user", "content": tool_results})

# Usage
if __name__ == "__main__":
    run_agentic_loop("Check if the repository has a test suite and run it.")