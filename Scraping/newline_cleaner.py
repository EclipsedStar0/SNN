import os
import re

def normalize_newlines(content):
    """
    Convert multiple consecutive newlines according to the rules:
    1 newline -> 0 newlines
    2 newlines -> 1 newline
    3 newlines -> 2 newlines
    4 newlines -> 3 newlines
    5 newlines -> 4 newlines
    More than 5 newlines -> 5 newlines (or 4 based on your spec)
    """
    # IMPORTANT: Process from smallest to largest to avoid cascading replacements
    # First replace single newlines with nothing
    content = re.sub(r'(?<!\n)\n(?!\n)', '', content)
    
    # Then replace double newlines with single newline
    content = content.replace('\n' * 2, '\n' * 1)
    
    # Then replace triple newlines with double newlines
    content = content.replace('\n' * 3, '\n' * 2)
    
    # Then replace quadruple newlines with triple newlines
    content = content.replace('\n' * 4, '\n' * 3)
    
    # Then replace quintuple newlines with quadruple newlines
    content = content.replace('\n' * 5, '\n' * 4)
    
    # Finally, for more than 5 newlines, replace with 4 newlines
    # (or 5 if you want to keep the pattern consistent)
    content = re.sub(r'\n{6,}', '\n' * 5, content)
    
    return content

def normalize_newlines_regex(content):
    """
    Alternative implementation using a single regex with callback function.
    This handles all cases at once without cascading issues.
    """
    def replace_match(match):
        newline_count = len(match.group(0))
        
        # Apply the rules:
        if newline_count == 1:
            return ''  # 0 newlines
        elif newline_count == 2:
            return '\n' * 1  # 1 newline
        elif newline_count == 3:
            return '\n' * 2  # 2 newlines
        elif newline_count == 4:
            return '\n' * 3  # 3 newlines
        elif newline_count == 5:
            return '\n' * 4  # 4 newlines
        else:  # 6 or more newlines
            return '\n' * 5  # 5 newlines (or 4 if you prefer)
    
    # Match 1 or more consecutive newlines
    return re.sub(r'\n+', replace_match, content)

def normalize_newlines_efficient(content):
    """
    Most efficient implementation using a single pass.
    This correctly handles all cases in one go.
    """
    result = []
    i = 0
    n = len(content)
    
    while i < n:
        # Count consecutive newlines
        newline_count = 0
        while i < n and content[i] == '\n':
            newline_count += 1
            i += 1
        
        if newline_count > 0:
            # Apply the transformation rules
            if newline_count == 1:
                # 1 newline -> 0 newlines (remove it)
                pass  # Don't add anything
            elif newline_count == 2:
                # 2 newlines -> 1 newline
                result.append('\n')
            elif newline_count == 3:
                # 3 newlines -> 2 newlines
                result.append('\n\n')
            elif newline_count == 4:
                # 4 newlines -> 3 newlines
                result.append('\n\n\n')
            elif newline_count == 5:
                # 5 newlines -> 4 newlines
                result.append('\n\n\n\n')
            else:
                # More than 5 newlines -> 5 newlines
                result.append('\n\n\n\n\n')
        else:
            # Add non-newline character
            result.append(content[i])
            i += 1
    
    return ''.join(result)

def process_file(filepath, method='efficient'):
    """Read, process, and overwrite a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process the content using selected method
        if method == 'efficient':
            processed_content = normalize_newlines_efficient(content)
        elif method == 'regex':
            processed_content = normalize_newlines_regex(content)
        else:
            processed_content = normalize_newlines(content)
        
        # Only write if content changed
        if content != processed_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            return True, "modified"
        else:
            return True, "unchanged"
    except Exception as e:
        return False, f"error: {e}"

def process_directory(directory_path, file_extensions=None):
    """
    Process all text files in the specified directory.
    
    Args:
        directory_path: Path to the directory to process
        file_extensions: List of file extensions to process (default: .txt files)
    """
    if file_extensions is None:
        file_extensions = ['.txt']
    
    # Convert extensions to lowercase for case-insensitive matching
    file_extensions = [ext.lower() for ext in file_extensions]
    
    # Get all files in the directory
    processed_count = 0
    modified_count = 0
    error_count = 0
    
    print(f"Processing directory: {directory_path}")
    print(f"File extensions to process: {file_extensions}")
    print("-" * 50)
    
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        
        # Check file extension
        _, ext = os.path.splitext(filename)
        if ext.lower() in file_extensions:
            print(f"Processing: {filename}", end="")
            
            success, status = process_file(filepath, method='efficient')
            
            if success:
                processed_count += 1
                if "modified" in status:
                    modified_count += 1
                print(f" - {status}")
            else:
                error_count += 1
                print(f" - ERROR: {status}")
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Files modified: {modified_count}")
    print(f"Files with errors: {error_count}")

def create_test_file():
    """Create a test file to demonstrate the transformation."""
    test_content = """Line 1

Line 2


Line 3



Line 4




Line 5





Line 6




Line 7"""
    
    with open("test_newlines.txt", "w") as f:
        f.write(test_content)
    
    print("Created test file: test_newlines.txt")
    print("\nOriginal content structure:")
    print(repr(test_content))
    
    # Test the transformation
    processed = normalize_newlines_efficient(test_content)
    print("\nProcessed content structure:")
    print(repr(processed))
    
    print("\nVisual comparison:")
    print("=" * 40)
    print("ORIGINAL:")
    print(test_content)
    print("=" * 40)
    print("PROCESSED:")
    print(processed)
    print("=" * 40)

def main():
    # Ask if user wants to create a test file first
    test_option = input("Create test file first to see the transformation? (yes/no): ").strip().lower()
    
    if test_option in ['yes', 'y']:
        create_test_file()
        print("\n" + "=" * 50 + "\n")
    
    # Get directory path from user or use current directory
    directory_path = input("Enter directory path to process (press Enter for current directory): ").strip()
    
    if not directory_path:
        directory_path = os.getcwd()
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    # Ask for file extensions
    extensions_input = input("Enter file extensions to process (comma-separated, default: .txt): ").strip()
    
    if extensions_input:
        file_extensions = [ext.strip() for ext in extensions_input.split(',')]
        # Ensure extensions start with a dot
        file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
    else:
        file_extensions = ['.txt']
    
    # Confirm with user
    print("\n" + "=" * 50)
    print(f"Directory: {directory_path}")
    print(f"File types: {file_extensions}")
    print("\nWARNING: This will OVERWRITE files in the directory!")
    print("Transformation rules:")
    print("  1 newline -> 0 newlines (removed)")
    print("  2 newlines -> 1 newline")
    print("  3 newlines -> 2 newlines")
    print("  4 newlines -> 3 newlines")
    print("  5 newlines -> 4 newlines")
    print("  6+ newlines -> 5 newlines")
    
    confirmation = input("\nDo you want to continue? (yes/no): ").strip().lower()
    
    if confirmation in ['yes', 'y']:
        process_directory(directory_path, file_extensions)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()