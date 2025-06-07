"""
Web Scrapers for Khmer News Sources

This module provides scrapers for major Khmer news websites with built-in
rate limiting, respectful crawling practices, and content deduplication.
"""

import requests
import time
import logging
import hashlib
import re
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime, timedelta
import random


@dataclass
class Article:
    """Data class for storing article information"""
    title: str
    content: str
    url: str
    date: Optional[str] = None
    author: Optional[str] = None
    source: str = ""
    content_hash: str = ""
    
    def __post_init__(self):
        """Generate content hash after initialization"""
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                (self.title + self.content).encode('utf-8')
            ).hexdigest()


class BaseScraper:
    """Base class for all news website scrapers"""
    
    def __init__(
        self, 
        base_url: str,
        name: str,
        delay_range: tuple = (1, 3),
        max_articles: int = 1000,
        timeout: int = 30
    ):
        self.base_url = base_url
        self.name = name
        self.delay_range = delay_range
        self.max_articles = max_articles
        self.timeout = timeout
        
        # Setup logging
        self.logger = logging.getLogger(f"scraper.{name}")
        
        # Setup session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'km,en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Track visited URLs and content hashes
        self.visited_urls: Set[str] = set()
        self.content_hashes: Set[str] = set()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_articles': 0,
            'failed_requests': 0,
            'duplicates_skipped': 0
        }
    
    def _respectful_delay(self):
        """Implement respectful delay between requests"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
        self.logger.debug(f"Waiting {delay:.2f} seconds before next request")
    
    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make a request with error handling and respect"""
        if url in self.visited_urls:
            self.stats['duplicates_skipped'] += 1
            self.logger.debug(f"Skipping already visited URL: {url}")
            return None
        
        self._respectful_delay()
        
        try:
            self.logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            self.visited_urls.add(url)
            self.stats['total_requests'] += 1
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            self.stats['failed_requests'] += 1
            self.logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
        
        # Normalize Unicode
        text = text.encode('utf-8').decode('utf-8')
        
        return text.strip()
    
    def _is_khmer_content(self, text: str, min_khmer_ratio: float = 0.6) -> bool:
        """Check if text contains sufficient Khmer content"""
        if not text:
            return False
        
        khmer_chars = sum(1 for char in text if '\u1780' <= char <= '\u17FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return False
        
        khmer_ratio = khmer_chars / total_chars
        return khmer_ratio >= min_khmer_ratio
    
    def _is_duplicate_content(self, content: str) -> bool:
        """Check if content is duplicate based on hash"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash in self.content_hashes:
            return True
        
        self.content_hashes.add(content_hash)
        return False
    
    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get list of article URLs from the website"""
        raise NotImplementedError("Subclasses must implement get_article_urls")
    
    def extract_article_content(self, url: str) -> Optional[Article]:
        """Extract article content from a given URL"""
        raise NotImplementedError("Subclasses must implement extract_article_content")
    
    def scrape_articles(self, max_articles: Optional[int] = None) -> List[Article]:
        """Main method to scrape articles from the website"""
        if max_articles is None:
            max_articles = self.max_articles
        
        self.logger.info(f"Starting scrape of {self.name} (max {max_articles} articles)")
        
        # Get article URLs
        article_urls = self.get_article_urls()
        self.logger.info(f"Found {len(article_urls)} article URLs")
        
        articles = []
        for i, url in enumerate(article_urls[:max_articles]):
            if len(articles) >= max_articles:
                break
            
            article = self.extract_article_content(url)
            if article:
                articles.append(article)
                self.stats['successful_articles'] += 1
                self.logger.info(f"Successfully scraped article {i+1}/{min(len(article_urls), max_articles)}")
            
            # Progress update every 10 articles
            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i+1}/{min(len(article_urls), max_articles)} articles processed")
        
        self.logger.info(f"Scraping completed. Collected {len(articles)} articles")
        self._log_statistics()
        
        return articles
    
    def _log_statistics(self):
        """Log scraping statistics"""
        self.logger.info("=== Scraping Statistics ===")
        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value}")


class VOAKhmerScraper(BaseScraper):
    """Scraper for VOA Khmer news website"""
    
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://www.voakhmernews.com",
            name="VOA_Khmer",
            **kwargs
        )
    
    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from VOA Khmer"""
        article_urls = []
        
        # Main sections to scrape
        sections = [
            "/section/cambodia",
            "/section/world", 
            "/section/politics",
            "/section/society"
        ]
        
        for section in sections:
            for page in range(1, max_pages + 1):
                section_url = f"{self.base_url}{section}?page={page}"
                soup = self._make_request(section_url)
                
                if not soup:
                    continue
                
                # Find article links
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    
                    # Filter for article URLs
                    if '/a/' in href and href not in article_urls:
                        full_url = urljoin(self.base_url, href)
                        article_urls.append(full_url)
                
                # Break if no more articles found
                if not links:
                    break
        
        return article_urls[:self.max_articles]
    
    def extract_article_content(self, url: str) -> Optional[Article]:
        """Extract article content from VOA Khmer URL"""
        soup = self._make_request(url)
        if not soup:
            return None
        
        try:
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            title = self._clean_text(title_elem.get_text()) if title_elem else ""
            
            # Extract content
            content_elem = soup.find('div', class_='article-content') or \
                          soup.find('div', class_='content') or \
                          soup.find('article')
            
            if not content_elem:
                return None
            
            # Remove unwanted elements
            for elem in content_elem.find_all(['script', 'style', 'nav', 'aside']):
                elem.decompose()
            
            content = self._clean_text(content_elem.get_text())
            
            # Validate content
            if not self._is_khmer_content(content) or self._is_duplicate_content(content):
                return None
            
            # Extract date
            date_elem = soup.find('time') or soup.find('span', class_='date')
            date = date_elem.get('datetime') if date_elem else None
            
            return Article(
                title=title,
                content=content,
                url=url,
                date=date,
                source="VOA_Khmer"
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None


class KhmerTimesScraper(BaseScraper):
    """Scraper for Khmer Times website"""
    
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://www.khmertimeskh.com",
            name="Khmer_Times",
            **kwargs
        )
    
    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from Khmer Times"""
        article_urls = []
        
        # Categories to scrape
        categories = [
            "/category/%e1%9e%80%e1%9e%98%e1%9f%92%e1%9e%96%e1%9e%bb%e1%9e%87%e1%9e%b6",  # Cambodia
            "/category/news",
            "/category/politics"
        ]
        
        for category in categories:
            for page in range(1, max_pages + 1):
                category_url = f"{self.base_url}{category}/page/{page}"
                soup = self._make_request(category_url)
                
                if not soup:
                    continue
                
                # Find article links
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    
                    # Filter for article URLs (year/month/day pattern)
                    if re.match(r'.*/\d{4}/\d{2}/\d{2}/.*', href) and href not in article_urls:
                        article_urls.append(href)
        
        return article_urls[:self.max_articles]
    
    def extract_article_content(self, url: str) -> Optional[Article]:
        """Extract article content from Khmer Times URL"""
        soup = self._make_request(url)
        if not soup:
            return None
        
        try:
            # Extract title
            title_elem = soup.find('h1', class_='entry-title') or soup.find('h1')
            title = self._clean_text(title_elem.get_text()) if title_elem else ""
            
            # Extract content
            content_elem = soup.find('div', class_='entry-content') or \
                          soup.find('div', class_='post-content')
            
            if not content_elem:
                return None
            
            # Remove unwanted elements
            for elem in content_elem.find_all(['script', 'style', 'nav', 'aside', 'figure']):
                elem.decompose()
            
            content = self._clean_text(content_elem.get_text())
            
            # Validate content
            if not self._is_khmer_content(content) or self._is_duplicate_content(content):
                return None
            
            # Extract date
            date_elem = soup.find('time', class_='entry-date')
            date = date_elem.get('datetime') if date_elem else None
            
            return Article(
                title=title,
                content=content,
                url=url,
                date=date,
                source="Khmer_Times"
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None


class RFAKhmerScraper(BaseScraper):
    """Scraper for RFA Khmer website"""
    
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://www.rfa.org/khmer",
            name="RFA_Khmer", 
            **kwargs
        )
    
    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from RFA Khmer"""
        article_urls = []
        
        # Sections to scrape
        sections = [
            "/news",
            "/politics", 
            "/society",
            "/business"
        ]
        
        for section in sections:
            section_url = f"{self.base_url}{section}"
            soup = self._make_request(section_url)
            
            if not soup:
                continue
            
            # Find article links
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                
                # Filter for article URLs
                if href.startswith('/khmer/') and '/news/' in href and href not in article_urls:
                    full_url = urljoin("https://www.rfa.org", href)
                    article_urls.append(full_url)
        
        return article_urls[:self.max_articles]
    
    def extract_article_content(self, url: str) -> Optional[Article]:
        """Extract article content from RFA Khmer URL"""
        soup = self._make_request(url)
        if not soup:
            return None
        
        try:
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            title = self._clean_text(title_elem.get_text()) if title_elem else ""
            
            # Extract content
            content_elem = soup.find('div', id='content') or \
                          soup.find('div', class_='article-content') or \
                          soup.find('div', class_='story-content')
            
            if not content_elem:
                return None
            
            # Remove unwanted elements
            for elem in content_elem.find_all(['script', 'style', 'nav', 'aside']):
                elem.decompose()
            
            content = self._clean_text(content_elem.get_text())
            
            # Validate content
            if not self._is_khmer_content(content) or self._is_duplicate_content(content):
                return None
            
            # Extract date
            date_elem = soup.find('span', class_='date') or soup.find('time')
            date = date_elem.get_text() if date_elem else None
            
            return Article(
                title=title,
                content=content,
                url=url,
                date=date,
                source="RFA_Khmer"
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None


# Factory function for easy scraper creation
def create_scraper(source: str, **kwargs) -> BaseScraper:
    """Factory function to create scrapers"""
    scrapers = {
        'voa': VOAKhmerScraper,
        'khmertimes': KhmerTimesScraper,
        'rfa': RFAKhmerScraper
    }
    
    if source.lower() not in scrapers:
        raise ValueError(f"Unknown source: {source}. Available: {list(scrapers.keys())}")
    
    return scrapers[source.lower()](**kwargs) 