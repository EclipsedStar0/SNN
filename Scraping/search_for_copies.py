
import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
from pathlib import Path


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
        "Which folder do you wish to search for duplicates in?",
        choices,
    )
    
    all_sub_folders = get_folder_names(folder_to_search)
    guten_ids = {}
    
    for sub_folder_name in all_sub_folders:
        files_in_folder = get_file_names(f'{folder_to_search}/{sub_folder_name}')
        for file_name in files_in_folder:
            if file_name.endswith(').txt'):
                t_str = file_name.split(').txt')[0]
                guten_id = t_str.split('(')[-1]
                if guten_id in guten_ids:
                    print(f"FOUND A COPY OF: ({file_name}) in {sub_folder_name}\n\tit is already present in {guten_ids[guten_id][0]} under the name {guten_ids[guten_id][1]}")
                else:
                    guten_ids[guten_id] = [sub_folder_name, file_name]