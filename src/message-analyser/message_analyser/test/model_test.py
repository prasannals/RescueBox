try:
    import ollama # type: ignore
except ImportError as e:
    print("The 'ollama' module could not be imported. Ensure it is installed in your environment.")
    raise e

response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": "Hello, how are you?"}])
print(response)