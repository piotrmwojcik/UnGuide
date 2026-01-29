#!/usr/bin/env python3
import os
import re
import sys

def fix_naming(directory):
    pattern = re.compile(r'^.+_(\d+)\.png$')

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = match.group(1)
            new_name = f"{number}.png"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            if old_path != new_path:
                print(f"{filename} -> {new_name}")
                os.replace(old_path, new_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    fix_naming(directory)
