#!/usr/bin/env python3
"""
Simple test script to verify uv-based PyRIT setup works correctly.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Test basic PyRIT imports
try:
    import pyrit
    print(f"✓ PyRIT version: {pyrit.__version__}")
except Exception as e:
    print(f"✗ Failed to import pyrit: {e}")
    sys.exit(1)

# Test core modules
modules_to_test = [
    "pyrit.memory",
    "pyrit.prompt_target",
    "pyrit.prompt_converter",
    "pyrit.score",
    "pyrit.chat_message_normalizer",
    "pyrit.common",
]

print("\nTesting module imports:")
for module in modules_to_test:
    try:
        __import__(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")

# Test specific classes
print("\nTesting key classes:")
test_imports = [
    ("pyrit.prompt_target", "OpenAIChatTarget"),
    ("pyrit.prompt_converter", "Base64Converter"),
    ("pyrit.score", "SelfAskTrueFalseScorer"),
]

for module_name, class_name in test_imports:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✓ {module_name}.{class_name}")
    except Exception as e:
        print(f"✗ {module_name}.{class_name}: {e}")

print("\n" + "="*50)
print("PyRIT setup verification complete!")
print("="*50)
