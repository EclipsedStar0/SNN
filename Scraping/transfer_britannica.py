"""
Web scraper that respects robots.txt and follows best practices.
"""

from typing import Any, TypeVar
import time
import re
import os
from pathlib import Path

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
    folder_name = "manual_download_backup"
    sub_folder_name = "Encyclopedia Britannica 11th Edition"
    file_names = get_file_names(f"{folder_name}/{sub_folder_name}/")
    for file_name in file_names:
        t_fname = file_name.split("_")
        s1_fname = t_fname[-1]
        s2_fname = "_".join(t_fname[0:-1])
        book_id = s1_fname.split(".")[0]
        extension = s1_fname.split(".")[-1]
        
        file_content = ""
        with open(f"{folder_name}/{sub_folder_name}/{file_name}", 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        with open(f"{folder_name}/{sub_folder_name}/{s2_fname} ({book_id}).{extension}", 'w', encoding='utf-8') as file:
            file.write(file_content)

if __name__ == "__main__":
    main()
