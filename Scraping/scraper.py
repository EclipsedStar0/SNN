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
    TARGET_URL = "https://www.gutenberg.org/browse/scores/top#books-last30"  # Example URL
    BASE_URL = "https://www.gutenberg.org"
    CONTENT_URL_PREFIX = "https://www.gutenberg.org"
    CONTENT_URL_SUFFIX = ".txt.utf-8"  # For plain text format
    
    folder_names = get_folder_names('')
    choices = []
    for folder_name in folder_names:
        choices.append(Choice(title=folder_name, value=folder_name))
    
    
    BASE_DIR = prompt_select("Select Root Dir for Storage", choices)
    print("\x1B[38;5;252m")
    OUTPUT_DIR = "Misc- Untested100"
    OUTPUT_DIR = f"{BASE_DIR}/{OUTPUT_DIR}"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Initialize scraper with 2-second delay between requests
    scraper = RespectfulScraper(BASE_URL, delay=2.0)
    
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
    
    print("\n" + "="*60)
    print("Step 1-3: Fetching and parsing target page")
    print("="*60)
    
    # Step 1-3: Navigate and get page content
    response = scraper.fetch_page(TARGET_URL)
    if not response:
        print("Failed to fetch the target page. Exiting.")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the div with class 'page_content'
    page_content_div = soup.find('div', class_='page_content')
    if not page_content_div:
        print("Could not find div with class 'page_content'")
        return
    
    print("\n" + "="*60)
    print("Step 4-5: Locating header and ordered list")
    print("="*60)
    
    # Step 4: Locate the header with id 'books-last1'
    header = page_content_div.find('h2', id='books-last30')
    if not header:
        print("Could not find header with id 'books-last30'")
        return
    
    print(f"✓ Found header: {header.get_text(strip=True)}")
    
    # Step 5: Find the ordered list after the header
    ol = header.find_next_sibling('ol')
    if not ol:
        print("Could not find ordered list after header")
        return
    
    print(f"✓ Found ordered list with {len(ol.find_all('li'))} items")
    
    print("\n" + "="*60)
    print("Step 6: Extracting links and titles")
    print("="*60)
    
    # Step 6: Extract links and text from list items
    link_catalog = {}
    
    for li in ol.find_all('li'):
        a_tag = li.find('a')
        if a_tag:
            href = a_tag.get('href')
            text = a_tag.get_text(strip=True)
            
            if href and text:
                link_spli = href.split("ebooks/")[-1]
                if link_spli in skip_ids:
                    continue
                
                tempp = text.split(" by ")
                if len(tempp) > 2:
                    t_name = tempp[-2]
                else:
                    t_name = tempp[0]
                
                link_catalog[link_spli] = t_name
                print(f"  • {text[:60]}...")
        
    print(f"\n✓ Extracted {len(link_catalog)} books")
    
    print("\n" + "="*60)
    print("Step 7: Segementing books by language...")
    print("="*60)
    
    rebuilt_catalog = {}
    for book_id in link_catalog:
        TARGET_URL = f'{BASE_URL}/ebooks/{book_id}'
        response = scraper.fetch_page(TARGET_URL)
        if not response:
            print("Failed to fetch the target page. Skipping.")
            continue
        
        cat_str = response.text.split("""<tr property="dcterms:language" datatype="dcterms:RFC4646" itemprop="inLanguage" content="en">""")[-1]
        cat_str = cat_str.split("<th>Category</th>")[-1]
        cat_str = cat_str.split('<td property="dcterms:type" datatype="dcterms:DCMIType">')[1]
        cat_str = cat_str.split("</td>")[0]
        if cat_str != 'Text':
            print('\tThis is not stored in .txt format')
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
    print("Step 8: Downloading books...")
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

if __name__ == "__main__":
    main()
