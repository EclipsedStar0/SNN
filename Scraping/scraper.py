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
            print(f"✓ robots.txt loaded from {robots_url}")
        except Exception as e:
            print(f"⚠ Warning: Could not read robots.txt: {e}")
            print("  Proceeding with caution...")
    
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
            print(f"✗ Blocked by robots.txt: {url}")
            return None
        
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            print(f"✓ Fetched: {url}")
            return response
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching {url}: {e}")
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
    OUTPUT_DIR = "scraped_content"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Initialize scraper with 2-second delay between requests
    scraper = RespectfulScraper(BASE_URL, delay=2.0)
    
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
    header = page_content_div.find('h2', id='books-last1')
    if not header:
        print("Could not find header with id 'books-last1'")
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
    links_array = []
    texts_array = []
    
    for li in ol.find_all('li'):
        a_tag = li.find('a')
        if a_tag:
            href = a_tag.get('href')
            text = a_tag.get_text(strip=True)
            
            if href and text:
                links_array.append(href)
                texts_array.append(text)
                print(f"  • {text[:60]}...")
    
    print(f"\n✓ Extracted {len(links_array)} books")
    
    print("\n" + "="*60)
    print("Step 7: Downloading content")
    print("="*60)
    
    # Step 7: Iterate over links and download content
    for i, (link, title) in enumerate(zip(links_array, texts_array), 1):
        print(f"\n[{i}/{len(links_array)}] Processing: {title[:50]}...")
        
        # Construct full URL for plain text content
        # Assuming the link is like "/ebooks/2701"
        # Convert to plain text URL
        book_id = link.split('/')[-1]
        full_url = f"{CONTENT_URL_PREFIX}/ebooks/{book_id}{CONTENT_URL_SUFFIX}"
        
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
                print(f"  ✓ Saved to: {filepath}")
            except Exception as e:
                print(f"  ✗ Error saving file: {e}")
        else:
            print(f"  ✗ Could not download content from {full_url}")
    
    print("\n" + "="*60)
    print("Scraping complete!")
    print(f"Files saved to: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
