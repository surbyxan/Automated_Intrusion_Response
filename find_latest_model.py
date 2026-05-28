import os
import re
import sys
import argparse

def find_latest_model(directory):
    """Finds the filename of the model with the highest step count in a directory."""
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return None
    
    # Try to find number in the standard format first: _step_NUMBER_steps
    pattern_preferred = re.compile(r'_step_(\d+)_steps')
    pattern_fallback = re.compile(r'(\d+)')
    
    files = [f for f in os.listdir(directory) if f.endswith('.zip')]
    
    model_data = []
    for f in files:
        match = pattern_preferred.search(f)
        if match:
            step_count = int(match.group(1))
        else:
            matches = pattern_fallback.findall(f)
            if matches:
                step_count = max(int(m) for m in matches)
            else:
                continue
        model_data.append((step_count, f))
    
    if not model_data:
        return None
    
    # Sort by step_count descending
    model_data.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(directory, model_data[0][1])

def main():
    parser = argparse.ArgumentParser(description='Find the latest model in a directory or verify a specific model path.')
    parser.add_argument('path', nargs='?', help='Directory to search for latest model or path to a specific model file.')
    parser.add_argument('--model', help='Explicitly provide a specific model file path.')

    args = parser.parse_args()
    
    # Use --model if provided, otherwise use the positional path argument
    target = args.model if args.model else args.path

    if not target:
        parser.print_help()
        sys.exit(1)

    if os.path.isfile(target):
        # If it's a file, just return the absolute path
        print(os.path.abspath(target))
    elif os.path.isdir(target):
        # If it's a directory, find the latest model
        latest = find_latest_model(target)
        if latest:
            print(os.path.abspath(latest))
        else:
            print(f"Error: No .zip models found in directory: {target}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Path does not exist or is invalid: {target}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
