import argparse
import json
import sys
from difflib import SequenceMatcher, unified_diff


def load_json(filepath):
    """
    Load a JSON file and return the parsed object.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        sys.exit(1)


def pretty_print_json(obj):
    """
    Return a pretty-printed JSON string with sorted keys.
    """
    return json.dumps(obj, indent=2, sort_keys=True)


def compute_similarity(text1, text2):
    """
    Compute a similarity ratio between two strings.
    """
    return SequenceMatcher(None, text1, text2).ratio()


def find_line_diffs(text1, text2, fromfile, tofile):
    """
    Generate a unified diff between two multi-line strings.
    """
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    return list(unified_diff(lines1, lines2, fromfile=fromfile, tofile=tofile, n=3))


def compare_json(obj1, obj2, path=""):  # noqa: C901
    """
    Recursively compare two JSON structures and collect differences.
    Returns a list of difference descriptions.
    """
    diffs = []

    if type(obj1) != type(obj2):
        diffs.append(f"Type mismatch at {path or 'root'}: {type(obj1).__name__} != {type(obj2).__name__}")
        return diffs

    # Compare dictionaries
    if isinstance(obj1, dict):
        keys1 = set(obj1.keys())
        keys2 = set(obj2.keys())
        for key in keys1 - keys2:
            diffs.append(f"Key '{key}' present in solution but missing in test at {path}")
        for key in keys2 - keys1:
            diffs.append(f"Key '{key}' present in test but missing in solution at {path}")
        for key in keys1 & keys2:
            new_path = f"{path}.{key}" if path else key
            diffs.extend(compare_json(obj1[key], obj2[key], new_path))

    # Compare lists
    elif isinstance(obj1, list):
        len1 = len(obj1)
        len2 = len(obj2)
        if len1 != len2:
            diffs.append(f"List length differs at {path}: solution has {len1} elements, test has {len2}")
        minlen = min(len1, len2)
        for i in range(minlen):
            new_path = f"{path}[{i}]"
            diffs.extend(compare_json(obj1[i], obj2[i], new_path))

    # Compare atomic values
    else:
        if obj1 != obj2:
            diffs.append(f"Value mismatch at {path}: solution={obj1!r} vs test={obj2!r}")

    return diffs


def main():
    parser = argparse.ArgumentParser(
        description="Compare a solution JSON file against a test JSON file and report differences."
    )
    parser.add_argument("solution_file", help="Path to the reference solution JSON file")
    parser.add_argument("test_file", help="Path to the JSON file to be tested")
    parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Show line-by-line unified diff of the two files"
    )
    args = parser.parse_args()

    # Load JSON files
    solution_json = load_json(args.solution_file)
    test_json = load_json(args.test_file)

    # Check direct equality
    equal = solution_json == test_json
    print(f"Test matches solution: {equal}")

    # Pretty-print and compute similarity
    text_sol = pretty_print_json(solution_json)
    text_test = pretty_print_json(test_json)
    ratio = compute_similarity(text_sol, text_test)
    print(f"Structural similarity: {ratio:.2%}")

    # Recursive semantic comparison
    diffs = compare_json(solution_json, test_json)
    if diffs:
        print(f"\nFound {len(diffs)} differences:")
        for diff in diffs:
            print(f"- {diff}")
    else:
        print("\nNo semantic differences found.")

    # Detailed line diff
    if args.detailed:
        print("\nUnified line-by-line diff (context=3):")
        for line in find_line_diffs(text_sol, text_test, args.solution_file, args.test_file):
            sys.stdout.write(line)


if __name__ == "__main__":
    main()