import os
import shutil

def duplicate_folder_structure(origin_path, destination_path, target_extension, replacement_extension, copy_content=False):
    """
    Duplicate folder structure with file handling based on specified parameters.
    
    Args:
    - origin_path (str): Root path of the origin directory
    - destination_path (str): Root path of the destination directory
    - target_extension (str): Extension of files to process (e.g., '.txt')
    - replacement_extension (str): New extension for duplicated files
    - copy_content (bool): Whether to copy file contents (default: False)
    """
    # Ensure extensions start with a dot
    tar_extension = ''
    if tar_extension != '':
        tar_extension = target_extension if target_extension.startswith('.') else '.' + target_extension
    
    replac_extension = ''
    if replac_extension != '':
        replac_extension = replacement_extension if replacement_extension.startswith('.') else '.' + replacement_extension
    
    # Walk through the origin directory
    for root, dirs, files in os.walk(origin_path):
        # Compute the corresponding destination subdirectory
        relative_path = os.path.relpath(root, origin_path)
        dest_root = os.path.join(destination_path, relative_path)
        
        # Create destination subdirectories
        os.makedirs(dest_root, exist_ok=True)
        
        # Process files
        for file in files:
            # Check if file matches target extension
            if target_extension == '' or file.lower().endswith(target_extension.lower()):
                # Construct full source and destination paths
                source_file = os.path.join(root, file)
                
                # Replace the extension
                spli_text = os.path.splitext(file)
                new_filename = spli_text[0] + replac_extension if replac_extension != '' else "".join(spli_text)
                dest_file = os.path.join(dest_root, new_filename)
                
                # Copy or create file
                if copy_content:
                    # Copy entire file contents
                    shutil.copy2(source_file, dest_file)
                else:
                    # Create an empty file
                    open(dest_file, 'a').close()
                
                print(f"Processed: {source_file} -> {dest_file}")

def main():
    # Example usage with input prompts
    origin_path = input("Enter the origin file path: ").strip()
    destination_path = input("Enter the destination file path: ").strip()
    target_extension = input("Enter the target file extension (e.g., .txt): ").strip()
    replacement_extension = input("Enter the replacement file extension (e.g., .bak): ").strip()
    copy_content_input = input("Copy file contents? (y/n): ").strip().lower()
    
    # Convert input to boolean
    copy_content = copy_content_input in ['y', 'yes', '1']
    print(target_extension)
    print(replacement_extension)
    
    
    # Validate paths
    if not os.path.exists(origin_path):
        print(f"Error: Origin path {origin_path} does not exist.")
        return
    
    # Create destination path if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)
    
    # Call the function to duplicate structure
    duplicate_folder_structure(
        origin_path, 
        destination_path, 
        target_extension, 
        replacement_extension, 
        copy_content
    )
    
    print("Folder structure duplication complete.")

if __name__ == "__main__":
    main()
