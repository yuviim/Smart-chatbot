# test_anthropic.py
import getpass
from anthropic import Anthropic

# Get API key directly
print("Please enter your Anthropic API key:")
api_key = getpass.getpass()

# Print the first few and last few characters (safely)
if len(api_key) > 10:
    print(f"API key starts with: {api_key[:4]}...")
    print(f"API key ends with: ...{api_key[-4:]}")
    print(f"API key length: {len(api_key)}")
else:
    print("WARNING: API key seems too short!")

# Create Anthropic client
client = Anthropic(api_key=api_key)

try:
    # Test the API with a simple message
    print("\nTesting API connection...")
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Hello! This is a test message."}
        ]
    )
    print("\nAPI test successful!")
    print(f"Response: {response.content}")
    
except Exception as e:
    print(f"\nAPI test failed with error: {e}")
    print("\nPossible issues:")
    print("1. The API key might be incorrect or expired")
    print("2. The model name might be incorrect")
    print("3. There might be network connectivity issues")
    print("4. Your account might have billing or access restrictions")
    
    # Check if key format looks correct
    if not api_key.startswith("sk-ant-"):
        print("\nNOTE: Anthropic API keys typically start with 'sk-ant-'")