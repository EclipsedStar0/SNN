import os
import re

import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
from pathlib import Path
from tqdm.auto import tqdm



def is_likely_poetry_line(line, next_line=None, prev_line=None, all_lines_indented=False, mode=''):
    """
    Detect if a line is likely part of poetry based on various heuristics.
    """
    # If ALL lines are indented (consistent formatting), it's NOT poetry
    # Poetry has VARIED indentation
    if mode=='Encyclopedia Britannica 11th Edition':
        return False
    
    
    if all_lines_indented:
        # Only flag as poetry if there's VARIED indentation
        if line and line[0] in ' \t':
            # Check if indentation is significantly different from neighbors
            current_indent = len(line) - len(line.lstrip())
            
            varied = False
            if prev_line:
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                if abs(current_indent - prev_indent) > 2:
                    varied = True
            if next_line:
                next_indent = len(next_line) - len(next_line.lstrip())
                if abs(current_indent - next_indent) > 2:
                    varied = True
            
            if varied:
                return True
        return False
    
    # Leading whitespace (indentation) is a poetry indicator
    # BUT only if not all lines are indented
    if line and line[0] in ' \t':
        return True
    
    # Short lines (under 60 chars) that don't end with sentence-ending punctuation
    # might be poetry, especially if followed by another short line
    if len(line.strip()) < 60 and next_line:
        if len(next_line.strip()) < 60:
            return True
    
    # Lines that end mid-phrase (with comma) and are VERY short (poetry-like)
    # Typical poetry lines are 20-50 chars, not 60-70 like wrapped prose
    if len(line.strip()) < 50 and line.rstrip().endswith(','):
        if next_line and len(next_line.strip()) < 50:
            return True
    
    return False


def is_section_header(line):
    """
    Detect if a line is a section header (title/heading).
    """
    stripped = line.strip()
    
    # Lines ending with commas or semicolons are NOT headers (they're list continuations)
    if stripped.endswith((',', ';')):
        return False
    
    # Lines with lots of commas are likely lists, not headers (3+ commas)
    if stripped.count(',') >= 3:
        return False
    
    # Short lines (< 80 chars) that are title case or all caps
    if len(stripped) < 80:
        # All caps (but not just abbreviations - must be a phrase)
        if stripped.isupper() and len(stripped) > 0:
            # If it's mostly single letters/abbreviations separated by commas, not a header
            if ',' in stripped:
                return False
            return True
        # Starts with capital and relatively short
        if stripped and stripped[0].isupper() and len(stripped) < 20:
            # Check if it's likely a title (no sentence-ending punctuation)
            if not stripped.endswith(('.', '!', '?')):
                return True
    
    return False


def is_field_label(line):
    """
    Detect if a line looks like a field label (e.g., "Budget:", "Chief of mission:", etc.)
    Uses patterns rather than hardcoded strings.
    """
    stripped = line.strip()
    
    if ':' not in stripped:
        return False
    
    colon_pos = stripped.index(':')
    
    # Very short labels (1-3 words before colon, under 30 chars)
    if colon_pos < 30:
        before_colon = stripped[:colon_pos].strip()
        word_count = len(before_colon.split())
        
        # 1-3 word phrases ending in colon are likely field labels
        if word_count <= 3:
            # But make sure it's not mid-sentence (like "Note: blah blah blah")
            # Field labels typically have capitalized words or are short
            if before_colon[0].isupper():
                return True
    
    return False


def line_starts_with_field_label(line):
    """
    Check if a line STARTS with a field label pattern (even if it has content after).
    E.g., "Budget: $5000" returns True
    """
    stripped = line.strip()
    
    if ':' not in stripped:
        return False
    
    colon_pos = stripped.index(':')
    
    # Check if the part before the colon looks like a field label
    if colon_pos < 30:
        before_colon = stripped[:colon_pos].strip()
        word_count = len(before_colon.split())
        
        if word_count <= 3 and before_colon and before_colon[0].isupper():
            return True
    
    return False


def should_join_lines(line1, line2, all_lines_indented=False, mode=''):
    """
    Determine if two consecutive lines should be joined.
    Returns True if they should be joined, False if they should remain separate.
    """
    # Don't join empty lines
    if not line1.strip() and not line2.strip():
        return False
    
    # Strip leading/trailing whitespace for analysis
    stripped1 = line1.strip()
    stripped2 = line2.strip()
    
    if mode == 'Encyclopedia Britannica 11th Edition' and len(stripped1) > 90 and not stripped1.endswith(('.', '!', '?', ':', '"', "'", ">", "}", "]", ")")):
        return True
    
    # Don't join if line2 starts with @ (field marker)
    if stripped2.startswith('@'):
        return False
    
    # Don't join if line2 is a section header
    if is_section_header(line2):
        return False
    
    # Don't join if line1 ends with colon (it's introducing a list or section)
    if stripped1.endswith(":"):
        return False
    
    if not is_section_header(stripped2) and ":" in stripped2:
        return False
    
    # Don't join if line2 starts with a field label pattern
    # (e.g., "Budget:", "Economic output:", etc.)
    if line_starts_with_field_label(line2):
        return False
    
    # === AGGRESSIVE JOINING RULES ===
    
    # Rule 1: Line ends mid-word/sentence (with lowercase letter)
    # This is the STRONGEST signal - always join
    if stripped1 and stripped1[-1].islower():
        return True
    
    # Rule 2: Line1 ends with semicolon (continuation in a list)
    if stripped1.endswith(';'):
        return True
    
    # Rule 3: Line1 ends with comma
    if stripped1.endswith(','):
        return True
    
    # Rule 4: Line1 doesn't end with strong punctuation and is reasonably long
    # Very likely a wrapped line - join it
    if len(stripped1) > 35 and not stripped1.endswith(('.', '!', '?', ':', '"', "'", ">", "}", "]", ")")):
        return True
    
    # Rule 5: Line1 ends with closing paren (often mid-sentence)
    if stripped1.endswith(')') and len(stripped1) > 30:
        return True
    
    return False


def normalize_ocr_text(text, mode=''):
    """
    Main function to normalize OCR text by intelligently joining lines.
    """
    lines = text.split('\n')
    
    # Detect if all non-empty lines are indented (common OCR artifact)
    non_empty_lines = [l for l in lines if l.strip()]
    if non_empty_lines:
        all_indented = all(line[0] in ' \t' for line in non_empty_lines if line)
    else:
        all_indented = False
    
    result = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i]
        
        # Handle empty lines - preserve them as paragraph breaks
        if not current_line.strip():
            result.append(current_line)
            i += 1
            continue
        
        # Get context for poetry detection
        prev_line = lines[i-1] if i > 0 else None
        next_line = lines[i+1] if i < len(lines) - 1 else None
        
        # Check if we're in a poetry section
        if mode != 'CIA Factbooks' and is_likely_poetry_line(current_line, next_line, prev_line, all_indented, mode):
            # Preserve poetry as-is
            result.append(current_line)
            i += 1
            continue
        
        # Look ahead and join lines if appropriate
        accumulated = current_line
        i += 1
        
        while i < len(lines):
            next_line = lines[i]
            
            # Stop at empty lines
            if not next_line.strip():
                break
            
            # Check if we should join
            if should_join_lines(accumulated, next_line, all_indented, mode):
                # Join with a space, preserving the content but normalizing whitespace
                accumulated = accumulated.rstrip() + ' ' + next_line.lstrip()
                i += 1
            else:
                # Don't join, break the loop
                break
        
        result.append(accumulated)
    
    return '\n'.join(result)

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
    content = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)
    
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
            return ' '  # 0 newlines
        elif newline_count == 2:
            return '\n' * 1  # 1 newline
        elif newline_count == 3:
            return '\n' * 1  # 2 newlines
        elif newline_count == 4:
            return '\n' * 2  # 3 newlines
        elif newline_count == 5:
            return '\n' * 2  # 4 newlines
        else:  # 6 or more newlines
            return '\n' * 3  # 5 newlines (or 4 if you prefer)
    
    # Match 1 or more consecutive newlines
    return re.sub(r'\n+', replace_match, content)
    

def normalize_newlines_regex_re(content):
    """
    Alternative implementation using a single regex with callback function.
    This handles all cases at once without cascading issues.
    """
    def replace_match(match):
        newline_count = len(match.group(0))
        
        # Apply the rules:
        if newline_count == 1:
            return ' '  # 0 newlines
        elif newline_count == 2:
            return '\n' * 1  # 1 newline
        elif newline_count == 3:
            return '\n' * 1  # 2 newlines
        elif newline_count == 4:
            return '\n' * 2  # 3 newlines
        elif newline_count == 5:
            return '\n' * 2  # 4 newlines
        else:  # 6 or more newlines
            return '\n' * 3  # 5 newlines (or 4 if you prefer)
    
    # Match 1 or more consecutive newlines
    return re.sub(r'\n+', replace_match, content.group())
    
def normalize_newlines_regex_keep_first(content):
    def replace_match(match):
        newline_count = len(match.group(0))
        
        # Apply the rules:
        if newline_count == 1:
            return '\n'  # 0 newlines
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
    
    
    
    

def normalize_white_space_regex(content):
    # Match 1 or more consecutive newlines
    return re.sub(r' +', ' ', content)
    
def normalize_tab_space_regex(content):
    return re.sub(r'\t ', '\t', content)
    
def normalize_tab_space_regex_p2(content):
    return re.sub(r' \t', '\t', content)
    

import ftfy

def normalize_newlines_efficient(content):
    """
    Most efficient implementation using a single pass.
    This correctly handles all cases in one go.
    """
    result = []
    i = 0
    content = ftfy.fix_text(content)
    n = len(content)
    while i < n-1:
        # Count consecutive newlines
        newline_count = 0
        # punctuation = ["'", '"', "?", "!", ".", ")", ">", "]", "}"]
        
        ended_with_punctuation = True
        
        # if i != 0 and content[i-1] in punctuation:
        # if content[i-1] == " ":
        #     ended_with_punctuation = False
        # 
        while i < n-1 and content[i] == '\n':
            newline_count += 1
            i += 1
        if newline_count > 0:
            if newline_count == 1:
                # 1 newline -> 0 newlines (remove it)
                result.append(' ')
            elif newline_count == 2:
                # 2 newlines -> 1 newline (remove it)
                result.append('\n')
            elif newline_count == 3:
                # 3 newlines -> 2 newlines
                result.append('\n')
            elif newline_count == 4:
                # 4 newlines -> 3 newlines
                result.append('\n\n')
            elif newline_count == 5:
                # 5 newlines -> 4 newlines
                result.append('\n\n')
            else:
                # More than 6 newlines -> 5 newlines
                result.append('\n\n\n')
                
        # if not ended_with_punctuation and newline_count > 0:
        #     if newline_count == 1:
        #         # 1 newline -> 0 newlines (remove it)
        #         result.append('\n')
        #     elif newline_count == 2:
        #         # 2 newlines -> 1 newline (remove it)
        #         result.append('\n')
        #     elif newline_count == 3:
        #         # 3 newlines -> 2 newlines
        #         result.append('\n\n')
        #     elif newline_count == 4:
        #         # 4 newlines -> 3 newlines
        #         result.append('\n\n\n')
        #     elif newline_count == 5:
        #         # 5 newlines -> 4 newlines
        #         result.append('\n\n\n\n')
        #     else:
        #         # More than 6 newlines -> 5 newlines
        #         result.append('\n\n\n\n\n')
                
        else:
            # Add non-newline character
            result.append(content[i])
            i += 1
        
    return ''.join(result)

def process_directory(storage_path, writing_path, mode, extensions):
    """
    Process all text files in the specified directory.
    
    Args:
        directory_path: Path to the directory to process
        file_extensions: List of file extensions to process (default: .txt files)
    """
    
    # Get all files in the directory
    processed_count = 0
    modified_count = 0
    error_count = 0
    
    print(f"Processing directory: {storage_path}")
    print(f"File extensions to process: {extensions}")
    print("-" * 50)
    
    sub_folder_names = get_folder_names(storage_path)
    Path(f"{writing_path}").mkdir(exist_ok=True)
    pattern = re.compile(r'\n*.{55,}\n\n.{40,}\n')
    for sub_folder_name in tqdm(sub_folder_names, desc=f"Processing Sub Folders..."):
        Path(f"{writing_path}/{sub_folder_name}").mkdir(exist_ok=True)
        file_names = get_file_names(f"{storage_path}/{sub_folder_name}")
        for file_name in tqdm(file_names, desc=f"Processing {sub_folder_name}..."):
            for extension in extensions:
                if file_name.endswith(extension):
                    file_content = ""
                    with open(f'{storage_path}/{sub_folder_name}/{file_name}', 'r', encoding='utf-8') as reading_file:
                        file_content = reading_file.read()
                    
                    
                    file_content = pattern.sub(normalize_newlines_regex_re, file_content)
                    file_content = normalize_white_space_regex(file_content)
                    file_content = normalize_tab_space_regex(file_content)
                    file_content = normalize_tab_space_regex_p2(file_content)
                    
                    
                    
                    
                    
                    
                    mode_to_use = ''
                    if sub_folder_name == 'CIA Factbooks' or sub_folder_name == 'Encyclopedia Britannica 11th Edition':
                        mode_to_use = sub_folder_name
                    
                    
                   
                    #     if sub_folder_name == 'Poetry' or sub_folder_name == 'Plays-Films-Dramas':
                    #         file_content = normalize_newlines_regex_keep_first(file_content)
                    #     else:    
                    #         if mode == 'efficient':
                    #             file_content = normalize_newlines_efficient(file_content)
                    #         elif mode == 'regex':
                    #             file_content = normalize_newlines_regex(file_content)
                    #         elif mode == 'standard':
                    #             file_content = normalize_newlines(file_content)
                    #
                    #if sub_folder_name == 'Encyclopedia Britannica 11th Edition':
                    #    file_content = normalize_newlines_regex(file_content) 
                    
                    
                    file_content = normalize_ocr_text(file_content, mode_to_use)
                        
                        
                    with open(f'{writing_path}/{sub_folder_name}/{file_name}', 'w', encoding='utf-8') as file:
                        file.write(file_content)
    print("File Modification Complete")



def prompt_select(message: str, choices: list[Any]) -> Any:
    return questionary.select(
            message,
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()

def get_file_names(directory_path, sort=True, exclude_hidden=False):
    """
    Get file names from a directory with options.
    
    Args:
        directory_path (str): Path to the directory
        sort (bool): Whether to sort file names alphabetically
        exclude_hidden (bool): Whether to exclude hidden files (starting with .)
    
    Returns:
        list: List of file names
    """
    try:
        # Convert to Path object for better handling
        path = Path(directory_path)
        
        # Check if path exists
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        
        # Check if it's a directory
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        # Get all files
        files = []
        for item in path.iterdir():
            if item.is_file():
                file_name = item.name
                # Skip hidden files if requested
                if exclude_hidden and file_name.startswith('.'):
                    continue
                files.append(file_name)
        
        # Sort if requested
        if sort:
            files.sort()
        
        return files
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_folder_names(directory_path, sort=True, exclude_hidden=False):
    """
    Get folder names from a directory with options.
    
    Args:
        directory_path (str): Path to the directory
        sort (bool): Whether to sort folder names alphabetically
        exclude_hidden (bool): Whether to exclude hidden folders (starting with .)
    
    Returns:
        list: List of folder names
    """
    try:
        # Convert to Path object for better handling
        path = Path(directory_path)
        
        # Check if path exists
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        
        # Check if it's a directory
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        # Get all directories
        folders = []
        for item in path.iterdir():
            if item.is_dir():
                folder_name = item.name
                # Skip hidden folders if requested
                if exclude_hidden and folder_name.startswith('.'):
                    continue
                folders.append(folder_name)
        
        # Sort if requested
        if sort:
            folders.sort()
        
        return folders
        
    except Exception as e:
        print(f"Error: {e}")
        return []
        

def main():
    # Ask if user wants to create a test file first
    choices = []
    folder_names = get_folder_names("")
    for folder_name in folder_names:
        choices.append(Choice(title=folder_name, value=folder_name))
    
    storage_location = prompt_select("Folder to Read From:", choices)
    writing_location = input("Folder to Write To: ")
    
    mode_selection = prompt_select("Select Normalization Mode",
        choices = [
            Choice(title='Efficient (Nope)', value='efficient'),
            Choice(title='Regex (use this)', value='regex'),
            Choice(title='Standard', value='standard'),
        ]
    )
    
    file_extensions = prompt_select("File Extension to Read From",
        choices = [
            Choice(title='TXT', value=['.txt']),
            Choice(title='IGNORE', value=['.IGNORE']),
            Choice(title='Both', value=['.txt', '.IGNORE']),
        ]
    )
    
    Path(writing_location).mkdir(exist_ok=True)
    
    print('Begining...')
    process_directory(storage_location, writing_location, mode_selection, file_extensions)

if __name__ == "__main__":
    main()