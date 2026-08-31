"""
Web scraper that respects robots.txt and follows best practices.
"""

import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import re
import os
from pathlib import Path
from tqdm.auto import tqdm

from bridges.bridges import *
from bridges.data_src_dependent import data_source
import sys

class RespectfulScraper:
    def __init__(self, base_url, delay=1.0):
        """
        Initialize the scraper with robots.txt compliance.
        
        Args:
            base_url: The base URL of the website
            delay: Delay between requests in seconds (default: 1.0)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; RespectfulBot/1.0; +http://example.com/bot)'
        })
        
        # Parse and check robots.txt
        self.robot_parser = RobotFileParser()
        robots_url = urljoin(base_url, '/robots.txt')
        self.robot_parser.set_url(robots_url)
        try:
            self.robot_parser.read()
            print(f"\x1B[38;5;82m✓\x1B[38;5;252m robots.txt loaded from {robots_url}")
        except Exception as e:
            print(f"⚠ Warning: Could not read robots.txt: {e}")
            print("  Proceeding with caution...")
    
    def can_fetch(self, url):
        """Check if we're allowed to fetch the URL according to robots.txt"""
        return self.robot_parser.can_fetch(self.session.headers['User-Agent'], url)
    
    def fetch_page(self, url, extra_indent=False):
        """
        Fetch a page with rate limiting and robots.txt compliance.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Response object or None if fetch failed
        """
        preffix = ""
        if extra_indent:
            preffix += "\t"
        
        if not self.can_fetch(url):
            print(f"{preffix}\x1B[38;5;160m✗\x1B[38;5;252m Blocked by robots.txt: {url}")
            return None
        
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            print(f"{preffix}\x1B[38;5;82m✓\x1B[38;5;252m Fetched: {url}")
            return response
        except requests.exceptions.RequestException as e:
            print(f"{preffix}\x1B[38;5;160m✗\x1B[38;5;252m Error fetching {url}: {e}")
            return None
    
    def sanitize_filename(self, filename):
        """Convert text to a safe filename"""
        # Remove/replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        # Trim and limit length
        filename = filename.strip()[:200]
        return filename if filename else 'untitled'


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
    # Configuration
    TARGET_URL = "https://www.gutenberg.org/browse/scores/top"  # Example URL
    BASE_URL = "https://www.gutenberg.org"
    CONTENT_URL_PREFIX = "https://www.gutenberg.org"
    CONTENT_URL_SUFFIX = ".txt.utf-8"  # For plain text format
    folder_names = get_folder_names('')
    choices = []
    for folder_name in folder_names:
        choices.append(Choice(title=folder_name, value=folder_name))
    
    
    BASE_DIR = prompt_select("Select Root Dir for Storage", choices)
    OUTPUT_DIR = input("\x1B[38;5;252mStorage Location: \x1B[38;5;208m")
    Use_OnHandList = prompt_select('Use on hand list', choices=[Choice(value=1, title='No; use Bookshelf'),Choice(value=2, title='No; Use Subjects'),  Choice(value=3, title='Yes')])
    
    OUTPUT_DIR = f"{BASE_DIR}/{OUTPUT_DIR}"

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    skip_ids = {}
        
    folder_to_search = BASE_DIR
    all_sub_folders = get_folder_names(folder_to_search)
    for sub_folder_name in all_sub_folders:
        files_in_folder = get_file_names(f'{folder_to_search}/{sub_folder_name}')
        for file_name in files_in_folder:
            t_str = file_name.split(').txt')[0]
            t_str = t_str.split(').IGNORE')[0]
            guten_id = t_str.split('(')[-1]
            skip_ids[guten_id] = True
    
    #print(skip_ids)
    scraper = RespectfulScraper(BASE_URL, delay=2.0)
    if Use_OnHandList != 3:
        srt = 'Bookshelf' if Use_OnHandList == 1 else 'Subject'
        sxt = 'bookshelf' if Use_OnHandList == 1 else 'subject'
        
        BOOKSHELF_ID = input(f"\x1B[38;5;252m{srt} ID: \x1B[38;5;208m")
        download_num = int(input("\x1B[38;5;252mNumber: \x1B[38;5;208m"))
        print("\x1B[38;5;252m")
        
        
        bookshelf_suffix = '?sort_order=downloads&start_index='
        index_num = 0
        skipped_books = 0
        
        # Initialize scraper with 2-second delay between requests
        
        link_catalog = {}
        

        for index in range(download_num // 25 + 1):
            TARGET_URL = f'{BASE_URL}/ebooks/{sxt}/{BOOKSHELF_ID}{bookshelf_suffix}{index*25+1}'
            
            print("\n" + "="*60)
            print(f"Step 1-3: Fetching and parsing target page: {TARGET_URL}")
            print("="*60)
            
            # Step 1-3: Navigate and get page content
            max_retries = 3
            while max_retries >= 0:
                response = scraper.fetch_page(TARGET_URL)
                if not response:
                    print("Failed to fetch the target page. Exiting.")
                    max_retries -= 1
                else:
                    max_retries = -1
            if max_retries != -1:
                continue
                
            
            s_str = response.text.split("""<span class="links">""")[1]
            t_str = s_str.split("</span>")[1:]
            t_str = "</span>".join(t_str)
            t_str = t_str.split("""<li class="statusline">""")[0]

            link_obj_arr = t_str.split("""\n<li class="booklink">\n""")
            
            passed_first = False
            
            for entry in link_obj_arr:
                if not passed_first:
                    passed_first =  True
                    continue
                if index_num >= download_num:
                    break
            
                # print(entry)
                a_et = entry.split("""<a class="link" href="/ebooks/""")[1]
                book_id = a_et.split("""" accesskey""")[0]
                if book_id in skip_ids:
                    skipped_books += 1
                    continue
                
                span_et = entry.split("""<span class="title">""")[1]
                title = span_et.split("</span>")[0]
                
                # print(span_et)
                span_et = entry.split("""<span class="subtitle">""")
                if len(span_et) < 2:
                    author = 'Various'
                else:
                    author = span_et[1].split("</span>")[0]
                link_catalog[book_id] = title+" by "+author
                index_num += 1
            
            print(link_catalog)
            print(f"\x1B[38;5;82m✓\x1B[38;5;252m Found {len(link_catalog)} possible candidate items")
            
            print("\n" + "="*60)
            
        rebuilt_catalog = {}
        for book_id in link_catalog:
            TARGET_URL = f'{BASE_URL}/ebooks/{book_id}'
            response = scraper.fetch_page(TARGET_URL)
            if not response:
                print("Failed to fetch the target page. Skipping.")
                continue
            s_str = response.text.split("""<tr property="dcterms:language" datatype="dcterms:RFC4646" itemprop="inLanguage" content="en">""")[-1]
            s_str = s_str.split("<th>Language</th>")[-1]
            s_str = s_str.split("<td>")[1]
            s_str = s_str.split("</td>")[0]
            
            if s_str.lower() == 'english':
                rebuilt_catalog[book_id] = ['.txt', f'{link_catalog.get(book_id)}']
            else:
                rebuilt_catalog[book_id] = ['.IGNORE', f'{link_catalog.get(book_id)}']
            
        index = 0
        for book_id in rebuilt_catalog:
            title = rebuilt_catalog.get(book_id)[1]
            print(f"\n[{index}/{len(rebuilt_catalog)}] Processing: {title}...")
            index += 1
            
            full_url = f"{CONTENT_URL_PREFIX}/files/{book_id}/{book_id}-8.txt"
            
            # Alternative: Try the .txt.utf-8 format or cache/epub directories
            content_response = scraper.fetch_page(full_url)
            
            if content_response and content_response.status_code == 200:
                # Sanitize filename
                safe_filename = scraper.sanitize_filename(title)
                filepath = os.path.join(OUTPUT_DIR, f"{safe_filename} ({book_id}){rebuilt_catalog[book_id][0]}")
                
                # Save content
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content_response.text)
                    print(f"  \x1B[38;5;82m✓\x1B[38;5;252m Saved to: {filepath}")
                except Exception as e:
                    print(f"  \x1B[38;5;160m✗\x1B[38;5;252m Error saving file: {e}")
                    
                    
                    
            else:
                print(f"  \x1B[38;5;160m✗\x1B[38;5;252m Could not download content from {full_url}")
                print('\t\tTrying with -0')
                
                full_url = f"{CONTENT_URL_PREFIX}/files/{book_id}/{book_id}-0.txt"
                content_response = scraper.fetch_page(full_url, True)
            
                if content_response and content_response.status_code == 200:
                    # Sanitize filename
                    safe_filename = scraper.sanitize_filename(title)
                    filepath = os.path.join(OUTPUT_DIR, f"{safe_filename} ({book_id}){rebuilt_catalog[book_id][0]}")
                    
                    # Save content
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content_response.text)
                        print(f"  \t\x1B[38;5;82m✓\x1B[38;5;252m Saved to: {filepath}")
                    except Exception as e:
                        print(f"  \t\x1B[38;5;160m✗\x1B[38;5;252m Error saving file: {e}")
                else:
                    print(f"  \x1B[38;5;160m✗\x1B[38;5;252m Could not download content from {full_url}")
                    print('\t\tTrying with nothing')
                    
                    full_url = f"{CONTENT_URL_PREFIX}/files/{book_id}/{book_id}.txt"
                    content_response = scraper.fetch_page(full_url, True)
                
                    if content_response and content_response.status_code == 200:
                        # Sanitize filename
                        safe_filename = scraper.sanitize_filename(title)
                        filepath = os.path.join(OUTPUT_DIR, f"{safe_filename} ({book_id}){rebuilt_catalog[book_id][0]}")
                        
                        # Save content
                        try:
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(content_response.text)
                            print(f"  \t\x1B[38;5;82m✓\x1B[38;5;252m Saved to: {filepath}")
                        except Exception as e:
                            print(f"  \t\x1B[38;5;160m✗\x1B[38;5;252m Error saving file: {e}")
                    else:
                        print(f"  \x1B[38;5;160m✗\x1B[38;5;252m Could not download content from {full_url}")
        
        print("\n" + "="*60)
        print("Scraping complete!")
        print(f"Files saved to: {OUTPUT_DIR}/")
        print("="*60)

    else:
        print("Enter/Paste your IDs. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            contents.append(line)
        attempt_ids = []
        for entry in contents:
            if entry == '': continue
            try:
                idd = int(entry)
                if entry not in skip_ids:
                    attempt_ids.append(idd)
            except TypeError:
                continue
        if len(attempt_ids) < 1:
            print(f"We don't have any targets; likely due to skips")
            return
        
        print('now reading..')
        unsort = ''
        with open(f'gutenburg_index.txt', 'r', encoding='utf-8') as r_file:
            unsort = r_file.read()
        
        catalog = {}
        for book_id in attempt_ids:
            position = unsort.find(f'   {book_id}\n')
            if position == -1:
                print(f'We could not find: {book_id}')
                continue
            
            nearby_string_start = max(0, position - 1024)
            nearby_string_end = max(0, position + 1024)
            subcat = unsort[nearby_string_start:nearby_string_end].split('\n\n')
            for entry in subcat:
                if entry.startswith(' '): continue
                top_index = 0
                split_by_nl = entry.split('\n')
                for index in range(len(split_by_nl)):
                    if split_by_nl[index] != '':
                        top_index = index
                        break
                acc_text = "\n".join(split_by_nl[0:len(split_by_nl)])
                top_len = len(split_by_nl[top_index])
                str_num = ""
                extension = '.txt'
                #print(split_by_nl[top_index])
                #print(f'Checking: [{split_by_nl[top_index][top_len-10:top_len]}]')
                
                for character in (split_by_nl[top_index][top_len-10:top_len]):
                    if character == '' or character == ' ': continue
                    
                    try:
                        temp = int(character)
                        str_num += character
                    except ValueError:
                        str_num = ''
                        break
                if str_num != '' and int(str_num) == book_id:
                    if '[Language: ' in acc_text and 'English' not in acc_text:
                        extension = '.IGNORE'
                    
                    catalog[int(str_num)] = f'{split_by_nl[top_index].split("   ")[0]} ({str_num}){extension}'
                
        proc_index = 1
        cat_len = len(catalog)
        possible_endings = ['-0.txt', '-8.txt', '.txt', '.txt.utf-8']
        for book_id in catalog:
            print(f"\n[{proc_index}/{cat_len}] Processing: {catalog.get(book_id).split(' (')[0]}...")
            #ensamble = str(book_id)
            #dir_link = []
            #if book_id >= 10:                
            #    for index in range(len(ensamble)-1):
            #        dir_link.append(ensamble[index])
            #else:
            #    dir_link.append('0')
            #dir_link.append(ensamble)
            #full_link = 'https://www.gutenberg.org/dirs/'+"/".join(dir_link)
            full_link = f'https://www.gutenberg.org/dirs/{book_id}'
            safe_filename = scraper.sanitize_filename(catalog.get(book_id))
            for ending in possible_endings:
                t_link = f'{full_link}/{book_id}{ending}'
                content_response = scraper.fetch_page(t_link, True)
                if content_response and content_response.status_code == 200:
                    try:
                        filepath = os.path.join(OUTPUT_DIR, f"{safe_filename}")
                        with open(filepath, 'w', encoding='utf-8') as w_file:
                            w_file.write(content_response.text)
                        print(f"  \t\x1B[38;5;82m✓\x1B[38;5;252m Saved to: {filepath}")
                        break
                    except Exception as e:
                        print(f"  \t\x1B[38;5;160m✗\x1B[38;5;252m Error saving file: {e}")
                else:
                    print(f"  \x1B[38;5;160m✗\x1B[38;5;252m Could not download content from {t_link}")
            proc_index += 1
        
        #bridges = Bridges(12864, "YOUR_USER_ID", "YOUR_API_KEY")
        #
        #print("Enter/Paste your IDs. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        #contents = []
        #while True:
        #    try:
        #        line = input()
        #    except EOFError:
        #        break
        #    contents.append(line)
        #attempt_ids = []
        #for entry in contents:
        #    if entry == '': continue
        #    try:
        #        idd = int(entry)
        #        if idd not in skip_ids:
        #            attempt_ids.append(idd)
        #    except TypeError:
        #        continue
        #for book_id in attempt_ids:
        #    print(book_id)
        #    meta = data_source.get_a_gutenberg_book_metadata(book_id)
        #    meta_auth = meta.authors
        #    meta_title = meta.title
        #    meta_lang = meta.lang
        #    
        #    authors = []
        #    for entry in meta_auth:
        #        a_str = entry.split(', ')
        #        a_str.reverse()
        #        a_str = ' '.join(a_str)
        #        authors.append(a_str)
        #    authors = ' and '.join(authors)
        #    title_str = f'{meta_title} by {authors}'
        #    
        #    extension = '.txt'
        #    if meta_lang != 'en':
        #        extension = '.IGNORE'
        #        
        #    safe_filename = scraper.sanitize_filename(title_str)
        #    filepath = os.path.join(OUTPUT_DIR, f"{safe_filename} ({book_id}){extension}")
        #    try:
        #        with open(filepath, 'w', encoding='utf-8') as w_file:
        #            w_file.write(data_source.gutenberg_book_text(book_id))
        #        print(f"  \t\x1B[38;5;82m✓\x1B[38;5;252m Saved to: {filepath}")
        #    except Exception as e:
        #        print(f"  \t\x1B[38;5;160m✗\x1B[38;5;252m Error saving file: {e}")
            

if __name__ == "__main__":
    main()
