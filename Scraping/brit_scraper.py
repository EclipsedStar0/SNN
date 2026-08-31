"""
Web scraper that respects robots.txt and follows best practices.
"""

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
            print(f"\x1B[38;5;252m✓ robots.txt loaded from {robots_url}")
        except Exception as e:
            print(f"\x1B[38;5;208m⚠ Warning: Could not read robots.txt: {e}")
            print("  Proceeding with caution...\x1B[38;5;252m")
    
    def can_fetch(self, url):
        """Check if we're allowed to fetch the URL according to robots.txt"""
        return self.robot_parser.can_fetch(self.session.headers['User-Agent'], url)
    
    def fetch_page(self, url):
        """
        Fetch a page with rate limiting and robots.txt compliance.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Response object or None if fetch failed
        """
        if not self.can_fetch(url):
            print(f"\x1B[38;5;160m✗ Blocked by robots.txt: {url}\x1B[38;5;252m")
            return None
        
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            print(f"\x1B[38;5;252m✓ Fetched: {url}")
            return response
        except requests.exceptions.RequestException as e:
            print(f"\x1B[38;5;160m✗ Error fetching {url}: {e}\x1B[38;5;252m")
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

def main():
    # Configuration
    TARGET_URL = "https://www.gutenberg.org/browse/scores/top"  # Example URL
    BASE_URL = "https://www.gutenberg.org"
    CONTENT_URL_PREFIX = "https://www.gutenberg.org"
    CONTENT_URL_SUFFIX = ".txt.utf-8"  # For plain text format
    OUTPUT_DIR = "britannica_scraped_content"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Initialize scraper with 2-second delay between requests
    scraper = RespectfulScraper(BASE_URL, delay=2.0)
    
    # print("\n" + "="*60)
    # print("Step 1-3: Fetching and parsing target page")
    # print("="*60)
    # 
    # # Step 1-3: Navigate and get page content
    # response = scraper.fetch_page(TARGET_URL)
    # if not response:
    #     print("Failed to fetch the target page. Exiting.")
    #     return
    # 
    # soup = BeautifulSoup(response.text, 'html.parser')
    # 
    # # Find the div with class 'page_content'
    # page_content_div = soup.find('div', class_='page_content')
    # if not page_content_div:
    #     print("Could not find div with class 'page_content'")
    #     return
    # 
    # print("\n" + "="*60)
    # print("Step 4-5: Locating header and ordered list")
    # print("="*60)
    # 
    # # Step 4: Locate the header with id 'books-last1'
    # header = page_content_div.find('h2', id='books-last1')
    # if not header:
    #     print("Could not find header with id 'books-last1'")
    #     return
    # 
    # print(f"✓ Found header: {header.get_text(strip=True)}")
    # 
    # # Step 5: Find the ordered list after the header
    # ol = header.find_next_sibling('ol')
    # if not ol:
    #     print("Could not find ordered list after header")
    #     return
    # 
    # print(f"✓ Found ordered list with {len(ol.find_all('li'))} items")
    # 
    # print("\n" + "="*60)
    # print("Step 6: Extracting links and titles")
    # print("="*60)
    # 
    # # Step 6: Extract links and text from list items
    # links_array = []
    # texts_array = []
    # 
    # for li in ol.find_all('li'):
    #     a_tag = li.find('a')
    #     if a_tag:
    #         href = a_tag.get('href')
    #         text = a_tag.get_text(strip=True)
    #         
    #         if href and text:
    #             links_array.append(href)
    #             texts_array.append(text)
    #             print(f"  • {text[:60]}...")
    
    ids = [
        34751,
        19846,
        40156,
        39232,
        39435,
        34992,
        38539,
        42048,
        34612,
        35561,
        40769,
        39632,
        41773,
        40863,
        32423,
        41343,
        43427,
        42638,
        37160,
        37736,
        34312,
        43254,
        41156,
        40641,
        32758,
        38799,
        37806,
        33698,
        35747,
        32975,
        34405,
        34702,
        38304,
        34018,
        19699,
        33750,
        30685,
        37523,
        39029,
        39127,
        38202,
        27478,
        27480,
        41902,
        38964,
        35306,
        42473,
        33189,
        34209,
        33614,
        41472,
        32689,
        33052,
        35169,
        33550,
        33239,
        34047,
        37880,
        30073,
        42854,
        34533,
        42552,
        35473,
        41567,
        33365,
        38892,
        35925,
        37610,
        32182,
        32783,
        27479,
        42736,
        36452,
        37984,
        33427,
        37282,
        31950,
        41685,
        39353,
        37064,
        31641,
        35398,
        31793,
        33127,
        34082,
        35606,
        42342,
        38401,
        38143,
        34162,
        31855,
        39908,
        35236,
        31447,
        37461,
        32860,
        38622,
        43060,
        31156,
        39775,
        34116,
        41264,
        41055,
        38454,
        42173,
        34878,
        40096,
        32607,
        39521,
        32940,
        40956,
        36226,
        33295,
        35092,
        40009,
        38709,
        32097,
        13600,
        39700,
        40538,
        40370,
        32294,
        36735,
        36104,
        30935,
        31329,
        32063,
        30976,
        33991,
        33477
    ]
    
    print(f"\n✓ Extracted {len(ids)} books")
    
    print("\n" + "="*60)
    print("Step 7: Downloading content")
    print("="*60)
    
    # Step 7: Iterate over links and download content
    index = 1
    for id in ids:
        title = f'Encylocopedia_Britannica_V11_{id}'
        print(f"\x1B[38;5;252m\n[{index}/{len(ids)}] Processing: {title}...")
        
        
        full_url = f"{CONTENT_URL_PREFIX}/files/{id}/{id}-8.txt"
        
        # Alternative: Try the .txt.utf-8 format or cache/epub directories
        content_response = scraper.fetch_page(full_url)
        
        if content_response and content_response.status_code == 200:
            # Sanitize filename
            safe_filename = scraper.sanitize_filename(title)
            filepath = os.path.join(OUTPUT_DIR, f"{safe_filename}.txt")
            
            # Save content
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content_response.text)
                print(f"\x1B[38;5;252m  ✓ Saved to: {filepath}")
            except Exception as e:
                print(f"\x1B[38;5;160m  ✗ Error saving file: {e}")
                
                
                
        else:
            print(f"\x1B[38;5;160m  ✗ Could not download content from {full_url}\x1B[38;5;252")
            print('\x1B[38;5;252mTrying with -0')
            
            full_url = f"{CONTENT_URL_PREFIX}/files/{id}/{id}-0.txt"
            content_response = scraper.fetch_page(full_url)
        
            if content_response and content_response.status_code == 200:
                # Sanitize filename
                safe_filename = scraper.sanitize_filename(title)
                filepath = os.path.join(OUTPUT_DIR, f"{safe_filename}.txt")
                
                # Save content
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content_response.text)
                    print(f"\x1B[38;5;252m  ✓ Saved to: {filepath}")
                except Exception as e:
                    print(f"\x1B[38;5;160m  ✗ Error saving file: {e}\x1B[38;5;252")
            else:
                print(f"\x1B[38;5;160m  ✗ Could not download content from {full_url}\x1B[38;5;252")
            
        index += 1
    
    print("\x1B[38;5;252\n" + "="*60)
    print("Scraping complete!")
    print(f"Files saved to: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
