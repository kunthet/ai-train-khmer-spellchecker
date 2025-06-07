#!/usr/bin/env python3
"""
Simple test script to validate web scraping setup
"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        from data_collection import create_scraper, TextProcessor, ContentDeduplicator
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_scraper_creation():
    """Test scraper creation"""
    try:
        from data_collection import create_scraper
        
        # Test creating each scraper type
        scrapers = ['voa', 'khmertimes', 'rfa']
        
        for scraper_name in scrapers:
            scraper = create_scraper(scraper_name, max_articles=5)
            print(f"✅ Created {scraper_name} scraper: {scraper.name} for {scraper.base_url}")
        
        return True
    except Exception as e:
        print(f"❌ Scraper creation error: {e}")
        return False

def test_text_processor():
    """Test text processing functionality"""
    try:
        from data_collection import TextProcessor
        
        processor = TextProcessor()
        
        # Test with sample Khmer text
        test_text = "<p>នេះជាអត្ថបទធេស្ត</p>"
        cleaned = processor.clean_text(test_text)
        print(f"✅ Text processing works: '{test_text}' -> '{cleaned}'")
        
        return True
    except Exception as e:
        print(f"❌ Text processor error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Khmer Web Scraping System")
    print("=" * 40)
    
    tests = [
        ("Module Imports", test_imports),
        ("Scraper Creation", test_scraper_creation),
        ("Text Processing", test_text_processor),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        result = test_func()
        results.append(result)
        print(f"Result: {'PASS' if result else 'FAIL'}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Summary: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Web scraping system is ready.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 