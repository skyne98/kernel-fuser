import inspect

import llama_cpp

print("llama_cpp version:", getattr(llama_cpp, "__version__", "unknown"))
print("llama_cpp file:", getattr(llama_cpp, "__file__", "unknown"))

print("\n--- Searching for 'tensor' related symbols ---")
for name in dir(llama_cpp):
    if "tensor" in name.lower():
        print(name)

print("\n--- Searching for 'get_model' related symbols ---")
for name in dir(llama_cpp):
    if "get_model" in name.lower():
        print(name)
