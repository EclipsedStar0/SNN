# Import necessary libraries
import matplotlib.pyplot as plt
import pickle
import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
from pathlib import Path
import math
import numpy
from numpy.polynomial.polynomial import Polynomial

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




# Define your array of numbers

data = []
data2 = []
data3 = []
avg_d = []


folder_names = get_folder_names("tracking/")
choices = []
for folder_name in folder_names:
    choices.append(Choice(title=folder_name, value=folder_name))

tracking_folder_name = prompt_select("Select Folder", choices)

file_names = get_file_names(f"tracking/{tracking_folder_name}/")
choices = []
for file_name in file_names:
    choices.append(Choice(title=file_name, value=file_name))
tracking_file_name = prompt_select("Select Tracking File", choices)

spliiit = tracking_file_name.split("_step_")
MODEL_NAME = spliiit[0]
step_num = spliiit[1]
step_num = step_num.split("_")[0]


view_choice = prompt_select("View Mode",
    choices = [
        Choice(title='Loss Over Time', value='lot'),
        Choice(title='Loss Change Over Time', value='lcot'),
        Choice(title='Wrong Tokens Over Time', value='alter'),
    ])

if view_choice == 'lot':

    #MODEL_NAME = 'HarvestMoon_V24K_Stage256'
    #model_suffix = ''

    #step_num = '4250'

    with open(f'tracking/{tracking_folder_name}/{MODEL_NAME}_step_{step_num}_tr_loss.pkl', 'rb') as file:
        data = pickle.load(file)
    with open(f'tracking/{tracking_folder_name}/{MODEL_NAME}_step_{step_num}_val_loss.pkl', 'rb') as file:
        data2 = pickle.load(file)
    with open(f'tracking/{tracking_folder_name}/{MODEL_NAME}_step_{step_num}_test_loss.pkl', 'rb') as file:
        data3 = pickle.load(file)
    with open(f'tracking/{tracking_folder_name}/{MODEL_NAME}_step_{step_num}_avg_tr_loss.pkl', 'rb') as file2:
        avg_d = pickle.load(file2)
    #print(data)


    print(data3)


    dat = ''
    for index in range(0, len(data), 10):
        dat += f'{data[index]:0.4f}, '

    datdat = [data[0]]
    prevStep = 1
    for index in range(len(data2)):
        for index2 in range(prevStep, data2[index][0]):
            datdat.append(data2[index][1])
        prevStep = data2[index][0]
        
    datdat2 = [data[0]]
    prevStep = 1
    for index in range(len(data3)):
        for index2 in range(prevStep, data3[index][0]):
            datdat2.append(data3[index][1])
        prevStep = data3[index][0]

    # Create a figure and axis
    plt.figure(figsize=(10, 5))

    # Plot the data
    plt.plot(data, marker='o', label='Training Loss', color='blue')
    plt.plot(avg_d, marker='x', label='AVG Training Loss', color='green')
    plt.plot(datdat, marker='^', label='VAL Training Loss', color='red')
    plt.plot(datdat2, marker='^', label='Test Training Loss', color='green')
    #plt.plot(graph_expan, marker='o', label='(IGNORE)', color='black')

    # Add titles and labels
    plt.title('Loss over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')

    # Show grid
    plt.grid()

    # Display the plot
    plt.show()
elif view_choice == 'lcot':
    with open(f'tracking/{tracking_folder_name}/{MODEL_NAME}_step_{step_num}_val_loss.pkl', 'rb') as file:
        data2 = pickle.load(file)
    
    skip_all_before_x = 0
    x_axis = []
    y_axis = []
    
    dmy_axis = []
    
    for index in range(len(data2)):
        #if index > 4:
        #    break
        if data2[index][0] > skip_all_before_x:
            break
        
        dmy_axis.append(-1)
    
    for index in range(len(dmy_axis), len(data2)):
        x_axis.append(data2[index-1][1]-data2[index][1])
        y_axis.append(data2[index][0])
    
    x_data = numpy.array(x_axis)
    y_data = numpy.array(y_axis)
    
    coefficients = Polynomial.fit(x_data, y_data, 3).convert().coef
    polynomial = Polynomial(coefficients)
    
    #x_fit = numpy.linspace(x_data[0] * 1.3, x_data[-1] * 20, len(x_data))
    x_fit = numpy.linspace(x_data[0], x_data[-1], 100)
    y_fit = polynomial(x_fit)
    
    
    print(x_axis)
    print(y_axis)
    
    # Create a figure and axis
    plt.figure(figsize=(10, 5))

    # Plot the data
    plt.plot(y_axis, x_axis, marker='o', label='Learning Improvement', color='blue')
    #plt.plot(y_fit, x_fit, marker='x', label='Line of Best Fit', color='red')

    # Add titles and labels
    plt.title('Rate of Learning')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Improvement')

    # Show grid
    plt.grid()

    # Display the plot
    plt.show()
elif view_choice == 'alter':
    with open(f'tracking/{tracking_folder_name}/{MODEL_NAME}_step_{step_num}_val_loss.pkl', 'rb') as file:
        data2 = pickle.load(file)
    
    skip_all_before_x = 100
    x_axis = []
    y_axis = []
    
    dmy_axis = []
    
    for index in range(len(data2)):
        #if index > 4:
        #    break
        if data2[index][0] > skip_all_before_x:
            break
        
        dmy_axis.append(-1)
    
    for index in range(len(dmy_axis), len(data2)):
        y_axis.append(pow(math.e, data2[index][1]))
        x_axis.append(data2[index][0])
    
    print(x_axis)
    print(y_axis)
    
    # Create a figure and axis
    plt.figure(figsize=(10, 5))

    # Plot the data
    plt.plot(x_axis, y_axis, marker='o', label='Learning Improvement', color='blue')
    plt.plot([x_axis[0]], [0], marker='x', label='ignore', color='green')

    # Add titles and labels
    plt.title('Rate of Learning')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Improvement')

    # Show grid
    plt.grid()

    # Display the plot
    plt.show()
    