import os
import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
from pathlib import Path
import re


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
        

def grab_file_names_recursive(directory_path):
    file_name_arr_temp = get_file_names(directory_path)
    file_name_arr = []
    for file_name in file_name_arr_temp:
        file_name_arr.append(f"{directory_path}/{file_name}")
    
    top_folders = get_folder_names(directory_path)
    for folder_name in top_folders:
        result = grab_file_names_recursive(f"{directory_path}/{folder_name}")
        if len(file_name_arr) > 0:
            file_name_arr.extend(result)
        elif len(result) > 0:
            file_name_arr = result
    
    return file_name_arr
    
def make_dires(file_name_array, base_path='', target_path='', target_ext='.txt', changed_ext='.TRA', copy_contents=False, ):
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    for file_name in file_name_array:
        # Make directories
        pot_folders = file_name.split("/")
        built_path = f'{target_path}'
        if len(pot_folders) > 1:
            for index in range(0, len(pot_folders)-1):
                built_path += f'/{pot_folders[index]}'
                os.makedirs(os.path.dirname(built_path), exist_ok=True)
        
        wout_exten = pot_folders[-1].split('.')[0]
        built_path += f'/{wout_exten}'
        if file_name.endswith(target_ext):
            if copy_contents:
                file_contents = ''
                with open(f'{target_path}/{file_name}', 'r', encoding='utf-8') as r_file:
                    file_contents = r_file.read()
                
                with open(f'{built_path}{changed_ext}', 'w', encoding='utf-8') as w_file:
                    w_file.write(file_contents)
            else:
                with open(f'{built_path}{changed_ext}', 'a', encoding='utf-8') as a_file:
                    pass
               
def quick_swap_slash(strr):
    return re.sub(r'\\', '/', strr)
    

if __name__ == "__main__":
    fetch_path_para = input('Initial Location: ')
    target_path_para = input('Target Destination: ')
    exten_target = input('Target .Ext (include .): ')
    new_exten = input('New .Ext (include .): ')
    copy_cont_flag = prompt_select('Copy file contents?', 
        choices = [
            Choice(title='No', value=False),
            Choice(title='Yes', value=True)
        ]
    )
    fetch_path_para = quick_swap_slash(fetch_path_para)
    target_path_para = quick_swap_slash(target_path_para)
    
    print('Grabbing list of files...')
    f_name_arr = grab_file_names_recursive(fetch_path_para)
    print(f_name_arr)
    print('Translating...')
    make_dires(f_name_arr, fetch_path_para, target_path_para, exten_target, new_exten, copy_cont_flag)
    print('Done!')