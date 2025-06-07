#!/usr/bin/env python3
"""
Demo Script for Khmer Web Scraping System

This script demonstrates how to use the web scraping system to collect
Khmer text data from news sources.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection import (
    VOAKhmerScraper, 
    KhmerTimesScraper, 
    RFAKhmerScraper,
    create_scraper,
    TextProcessor,
    ContentDeduplicator
)


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup file handler
    log_filename = f"logs/scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def save_articles_to_file(articles, filename):
    """Save articles to a text file"""
    os.makedirs('output', exist_ok=True)
    filepath = f"output/{filename}"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Khmer Text Corpus - {len(articles)} articles\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for i, article in enumerate(articles, 1):
            f.write(f"Article {i}: {article.title}\n")
            f.write(f"Source: {article.source}\n")
            f.write(f"URL: {article.url}\n")
            f.write(f"Date: {article.date}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{article.content}\n")
            f.write("\n" + "=" * 60 + "\n\n")
    
    return filepath


def demo_single_scraper(scraper_name, max_articles=10):
    """Demonstrate single scraper functionality"""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Demo: {scraper_name} Scraper ===")
    
    try:
        # Create scraper
        scraper = create_scraper(
            scraper_name, 
            max_articles=max_articles,
            delay_range=(1, 2)  # Faster for demo
        )
        
        logger.info(f"Created {scraper_name} scraper")
        
        # Scrape articles
        articles = scraper.scrape_articles(max_articles=max_articles)
        
        if articles:
            logger.info(f"Successfully scraped {len(articles)} articles")
            
            # Process articles
            processor = TextProcessor()
            processed_articles = processor.process_articles(articles)
            
            logger.info(f"After processing: {len(processed_articles)} articles")
            
            # Save results
            filename = f"{scraper_name}_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = save_articles_to_file(processed_articles, filename)
            logger.info(f"Articles saved to: {filepath}")
            
            return processed_articles
        else:
            logger.warning("No articles were scraped")
            return []
            
    except Exception as e:
        logger.error(f"Error in {scraper_name} demo: {e}")
        return []


def demo_multi_source_scraping(max_articles_per_source=20):
    """Demonstrate multi-source scraping with deduplication"""
    logger = logging.getLogger(__name__)
    logger.info("=== Demo: Multi-Source Scraping with Deduplication ===")
    
    all_articles = []
    sources = ['voa', 'khmertimes', 'rfa']
    
    for source in sources:
        logger.info(f"Scraping from {source}...")
        
        try:
            scraper = create_scraper(
                source,
                max_articles=max_articles_per_source,
                delay_range=(1, 2)
            )
            
            articles = scraper.scrape_articles(max_articles=max_articles_per_source)
            
            if articles:
                all_articles.extend(articles)
                logger.info(f"Collected {len(articles)} articles from {source}")
            else:
                logger.warning(f"No articles collected from {source}")
                
        except Exception as e:
            logger.error(f"Error scraping {source}: {e}")
            continue
    
    if not all_articles:
        logger.error("No articles collected from any source")
        return []
    
    logger.info(f"Total articles before processing: {len(all_articles)}")
    
    # Process articles
    processor = TextProcessor()
    processed_articles = processor.process_articles(all_articles)
    
    logger.info(f"After text processing: {len(processed_articles)} articles")
    
    # Deduplicate
    deduplicator = ContentDeduplicator()
    unique_articles = deduplicator.deduplicate_articles(processed_articles)
    
    logger.info(f"After deduplication: {len(unique_articles)} articles")
    
    # Save results
    filename = f"multi_source_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = save_articles_to_file(unique_articles, filename)
    logger.info(f"Final corpus saved to: {filepath}")
    
    return unique_articles


def demo_text_processing():
    """Demonstrate text processing capabilities"""
    logger = logging.getLogger(__name__)
    logger.info("=== Demo: Text Processing ===")
    
    # Sample dirty text
    dirty_text = """
    <h1>ព័ត៌មានថ្មី</h1>
    <p>នេះជាអត្ថបទ (test article) ដែលមានបញ្ហា HTML tags និង   
    whitespace ច្រើនពេក។</p>
    
    [ភ្នាក់ងារ] - មានតំណភ្ជាប់: http://example.com និង email: test@example.com
    
    Phone: +855 12 345 678
    !!!????
    
    Unicode issues: â€œquotesâ€
    """
    
    logger.info("Original text:")
    logger.info(repr(dirty_text))
    
    # Process text
    processor = TextProcessor()
    
    # Step by step processing
    step1 = processor.clean_html_artifacts(dirty_text)
    logger.info("After HTML cleaning:")
    logger.info(repr(step1))
    
    step2 = processor.normalize_unicode_text(step1)
    logger.info("After Unicode normalization:")
    logger.info(repr(step2))
    
    step3 = processor.remove_unwanted_content(step2)
    logger.info("After unwanted content removal:")
    logger.info(repr(step3))
    
    final = processor.normalize_whitespace(step3)
    logger.info("Final cleaned text:")
    logger.info(repr(final))
    
    # Calculate quality metrics
    metrics = processor.calculate_quality_metrics(final)
    logger.info(f"Quality metrics: {metrics}")


def main():
    """Main demo function"""
    # Setup logging
    logger = setup_logging(logging.INFO)
    logger.info("Starting Khmer Web Scraper Demo")
    
    try:
        # Demo 1: Text processing
        demo_text_processing()
        
        # Demo 2: Single scraper (small sample for testing)
        logger.info("\n" + "="*60)
        demo_single_scraper('voa', max_articles=5)
        
        # Demo 3: Multi-source scraping (uncomment for full demo)
        # logger.info("\n" + "="*60)
        # demo_multi_source_scraping(max_articles_per_source=10)
        
        logger.info("Demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 