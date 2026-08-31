import datasets
import numpy
import pickle
import os
from tqdm.auto import tqdm

num_in_tol = 0

# Load the dataset from disk
ds = datasets.load_from_disk("hf_dataset/LoC-PD-Books")
marker = [0, 200, 1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 150000, 200000, 300000, 500000, 1000000]
counter = []
indexes = {}

for entry in marker:
    counter.append(0)
    indexes[entry] = []

total_entries = len(ds['train'])
print(f'Total Number of Entries: {total_entries}')

progress_bar = tqdm(total=total_entries, desc="Scanning books...")
for index in range(total_entries):
    entry = ds['train'][index]['text'].split()
    lenI = len(entry)
    for index2 in range(len(marker)):
        if index2 == len(marker)-1 and lenI >= marker[index2]:
            counter[-1] += 1
            indexes[marker[index2]].append(index)
        else:
            if lenI >= marker[index2] and lenI < marker[index2+1]:
                counter[index2] += 1
                indexes[marker[index2]].append(index)
    ds['train'][index]['text'] = ''
    progress_bar.update(1)  # Update the progress bar
    

print(f"""Books between 0-200 words: {counter[0]}
Books between 200-1000 words: {counter[1]}
Books between 1000-5000 words: {counter[2]}
Books between 5000-10000 words: {counter[3]}
Books between 10000-20000 words: {counter[4]}
Books between 20000-40000 words: {counter[5]}
Books between 40000-60000 words: {counter[6]}
Books between 60000-80000 words: {counter[7]}
Books between 80000-100000 words: {counter[8]}
Books between 100000-200000 words: {counter[9]}
Books between 200000-300000 words: {counter[10]}
Books between 300000-500000 words: {counter[11]}
Books between 500000-1000000 words: {counter[12]}
Books between 1000000+ words: {counter[13]}""")

DMP_NAME = 'words_sl1'

os.makedirs(os.path.dirname(f"dataset_len_dumps/"), exist_ok=True)
with open(f'dataset_len_dumps/{DMP_NAME}.pkl', 'wb') as file:
    pickle.dump(indexes, file)