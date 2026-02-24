#!/usr/bin/env python
"""Launcher script for Wine Python - adds correct path and runs target script"""
import sys
import os

# Add the production directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Get the target script from command line
if len(sys.argv) < 2:
    print("Usage: launcher.py <script.py> [arguments...]")
    sys.exit(1)

target_script = sys.argv[1]
sys.argv = sys.argv[1:]  # Remove launcher.py from argv

# Execute the target script
target_path = os.path.join(script_dir, target_script)
with open(target_path, 'r', encoding='utf-8') as f:
    code = f.read()
exec(compile(code, target_path, 'exec'))
