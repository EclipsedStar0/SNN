import pickle
import datasets
import re
ds = datasets.load_from_disk("hf_dataset/LoC-PD-Books")
indexes = {}


with open(f'dataset_len_dumps/words_sl1.pkl', 'rb') as file:
    indexes = pickle.load(file)
markers = [0, 200, 1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 150000, 200000, 300000, 500000, 1000000]
fetch_count = [0, 0, 10000, 5000, 2500, 1000, 100, 50, 40, 30, 20, 10, 8, 4]  # Adjust this array for your needs

# Initialize an array to store the fetched texts
fetched_books = []

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
            return '\n' * 3  # 4 newlines
        else:  # 6 or more newlines
            return '\n' * 3  # 5 newlines (or 4 if you prefer)
    
    # Match 1 or more consecutive newlines
    return re.sub(r'\n+', replace_match, content)

total_words_added = 0
total_books_added = 0

# Iterate through the markers and fetch the specified number of books
for word_range_index, count in enumerate(fetch_count):
    if count > 0:
        book_indices = indexes[markers[word_range_index]]
        for i in range(min(count, len(book_indices))):
            book_index = book_indices[i]
            entry_data = ds['train'][book_index]['text']
            entry_data = normalize_newlines_regex(entry_data)
            fetched_books.append(entry_data)
            total_books_added += 1
            total_words_added += len(entry_data.split())
            
print(f"Added {total_books_added} books from LoC for a total of {total_words_added} additional words")

# At this point, fetched_books contains the specified texts
# print(f"Fetched {len(fetched_books)} books:")
# for i, text in enumerate(fetched_books):
#     print(f"Book {i+1} Text: [{text[10000:10600]}]...")  # Print third 100 characters