import re
import sqlite3
import os
import ftfy

def count_adjectives_batch(my_data, adjectives_file_path, db_file_path, batch_size=100):
    """
    Process adjectives in batches for better performance with large files.
    """
    try:
        # Read adjectives
        with open(adjectives_file_path, 'r') as file:
            adjectives = [line.strip().lower() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {adjectives_file_path} not found.")
        return
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS FCT_Adjectives (
                word_id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                matches INTEGER NOT NULL
            )
        ''')
        
        # Clear existing data
        cursor.execute("DELETE FROM FCT_Adjectives")
        
        # Process in batches
        for i in range(0, len(adjectives), batch_size):
            batch = adjectives[i:i + batch_size]
            batch_data = []
            
            for adjective in batch:
                pattern = re.compile(re.escape(adjective), re.IGNORECASE)
                matches = len(pattern.findall(my_data))
                batch_data.append((adjective, matches))
            
            # Insert batch
            cursor.executemany(
                "INSERT INTO FCT_Adjectives (word, matches) VALUES (?, ?)",
                batch_data
            )
            
            print(f"Processed batch {i//batch_size + 1}/{(len(adjectives)-1)//batch_size + 1}")
        
        conn.commit()
        print(f"Successfully processed {len(adjectives)} adjectives.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

# Example usage for batch processing
if __name__ == "__main__":
    my_data = "Your large string data here..."
        # Sample data
    training_data = [
        'cats rule the world.',
        'dogs are the best.', 
        'elephants have long trunks.',
        'monkeys like bananas.',
        'pandas eat bamboo.',
        'tigers are dangerous.',
        'zebras have stripes.',
        'lions are the kings of the savannah.',
        'giraffes have long necks.',
        'hippos are big and scary.',
        'rhinos have horns.',
        'penguins live in the arctic.',
        'polar bears are white.',
        '\n——————————————————————\n'
    ]
    
    # Add your file data
    files_to_load = [
        "data/dominion_rp_epd.txt",
        "data/dominion_rp_disestro.txt",
        "data/dominion_rp_electua_solo_only.txt",
        "data/worm_mini_non_fanfic.txt",
        "data/sierra_data.txt", 
        "data/forsaken_data.txt",
        "data/short_snippets.txt",
        "data/additional_stories.txt",
        "data/merek_vr_ravenfield.txt",
        "data/mini_litrpg_abomination.txt",
        "data/basic_world_knowledge.txt",
    ]

    for file_path in files_to_load:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                training_data.append(content)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            continue
    
    training_data = [ftfy.fix_text(text) for text in training_data]
    training_data = " ".join(training_data)
    
    
    count_adjectives_batch(
        my_data=training_data,
        adjectives_file_path="data/adjectives.txt",
        db_file_path="data/word_data.db",
        batch_size=100
    )