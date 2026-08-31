
import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
from pathlib import Path
import ftfy
from tqdm.auto import tqdm

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
        
        

if __name__ == "__main__":
    folder_names = get_folder_names('')
    choices = []
    for name in folder_names:
        choices.append(Choice(title=name, value=name))
    
    
    folder_to_search = prompt_select(
        "Which folder do you wish to count?",
        choices,
    )
    
    all_sub_folders = get_folder_names(folder_to_search)
    
    total_characters = 0
    total_words = 0
    total_works = 0
    
    words_for_folders = {}
    
    top_f_names = {}
    
    for sub_folder_name in tqdm(all_sub_folders, desc='Scanning folders...'):
        words_for_folders[sub_folder_name] = 0
        files_in_folder = get_file_names(f'{folder_to_search}/{sub_folder_name}')
        for file_name in tqdm(files_in_folder, desc=f'Scanning files in {sub_folder_name}'):
            if file_name.endswith('.txt'):
                total_works += 1
                with open(f'{folder_to_search}/{sub_folder_name}/{file_name}', 'r', encoding="utf-8") as file:
                    charas = file.read()
                    charas = ftfy.fix_text(charas)
                    total_characters += len(charas)
                    num_w = len(charas.split())
                    words_for_folders[sub_folder_name] += num_w
                    total_words += num_w
        t_name = sub_folder_name.split("- ")
        if len(t_name) > 1:
            if t_name[0] not in top_f_names:
                top_f_names[t_name[0]] = {
                    'Categories': [],
                    'WCount': 0,
                }
            top_f_names[t_name[0]]['Categories'].append(sub_folder_name)
            top_f_names[t_name[0]]['WCount'] += words_for_folders.get(sub_folder_name)
        else:
            top_f_names[sub_folder_name] = {
                    'Categories': [sub_folder_name],
                    'WCount': words_for_folders.get(sub_folder_name),
                }
            
    
    print(f"Total Books Scanned: {total_works:,}")
    print(f"- Total Words: {total_words:,}")
    print(f"- Total Characters: {total_characters:,}")
    print()
    
    for top_folder in top_f_names:
        if top_f_names[top_folder]['WCount'] == 0:
            continue
        print(f"{top_folder}: {top_f_names[top_folder]['WCount']:,}, ({100*top_f_names[top_folder]['WCount']/total_words:.2f}%)")
        for sub_folder in top_f_names[top_folder]['Categories']:
            if words_for_folders[sub_folder] == 0:
                continue
            
            print(f"\t{sub_folder}: {words_for_folders[sub_folder]:,}, ({100*words_for_folders[sub_folder]/top_f_names[top_folder]['WCount']:.2f}%)")
    