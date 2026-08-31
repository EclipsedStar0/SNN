import os
import re

import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
from pathlib import Path
from tqdm.auto import tqdm

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
    def replace_match(match):
        ws_count = len(match.group(0))
        
        # Apply the rules:
        if ws_count > 0:
            return " "
    
    # Match 1 or more consecutive newlines
    return re.sub(r'\s', replace_match, content)

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
                result.append('\n\n')
            elif newline_count == 4:
                # 4 newlines -> 3 newlines
                result.append('\n\n\n')
            elif newline_count == 5:
                # 5 newlines -> 4 newlines
                result.append('\n\n\n\n')
            else:
                # More than 6 newlines -> 5 newlines
                result.append('\n\n\n\n\n')
                
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
    for sub_folder_name in tqdm(sub_folder_names, desc=f"Processing Sub Folders..."):
        Path(f"{writing_path}/{sub_folder_name}").mkdir(exist_ok=True)
        file_names = get_file_names(f"{storage_path}/{sub_folder_name}")
        for file_name in tqdm(file_names, desc=f"Processing {sub_folder_name}..."):
            for extension in extensions:
                if file_name.endswith(extension):
                    file_content = ""
                    with open(f'{storage_path}/{sub_folder_name}/{file_name}', 'r', encoding='utf-8') as reading_file:
                        file_content = reading_file.read()

                    if mode == 'efficient':
                        file_content = normalize_newlines_efficient(file_content)
                    elif mode == 'regex':
                        file_content = normalize_newlines_regex(file_content)
                    elif mode == 'standard':
                        file_content = normalize_newlines(file_content)
                    
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