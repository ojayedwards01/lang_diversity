import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
import re
import warnings

# Set seed for reproducible results
DetectorFactory.seed = 42

warnings.filterwarnings('ignore')

def clean_text(text):
    """Clean text by removing URLs, special characters, and extra whitespace"""
    if pd.isna(text) or text == '':
        return ''
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\\w\\s\\.\\,\\!\\?\\;\\:\\-\\(\\)\\[\\]\\{\\}]', '', str(text))
    
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', str(text)).strip()
    
    return text

def detect_language_langdetect(text):
    """Detect language using langdetect with error handling"""
    try:
        return detect(text)
    except:
        return 'unknown'

def analyze_languages_with_langdetect(csv_file, language_name):
    """Analyze CSV file using langdetect for language detection"""
    print(f"\\n{'='*60}")
    print(f"LANGDETECT ANALYSIS: {language_name.upper()}")
    print(f"{'='*60}")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} posts from {csv_file}")
    
    # Prepare results data
    results_data = []
    
    # Process each post
    for idx, row in df.iterrows():
        # content = row['content'] if pd.notna(row['content']) else '' # For Reddit Datasets
        content = row['Text'] if pd.notna(row['Text']) else ''
        
        if content and len(content.strip()) > 10:  # Only process non-empty content
            # Clean the text
            cleaned_content = clean_text(content)
            
            if cleaned_content:
                try:
                    # Get langdetect prediction
                    predicted_language = detect_language_langdetect(cleaned_content)
                    
                    # langdetect doesn't provide confidence scores, so we'll use a default
                    confidence_score = 1.0  # langdetect is deterministic
                    
                    results_data.append({
                        'content': cleaned_content,
                        'predicted_language': predicted_language,
                        'confidence_score': confidence_score
                    })
                    
                    # Print progress
                    print(f"Post {idx+1}: {cleaned_content[:100]}...")
                    print(f"  → Predicted: {predicted_language}")
                    print()
                    
                except Exception as e:
                    print(f"Error processing post {idx+1}: {e}")
                    continue
    
    # Save results to CSV
    if results_data:
        results_df = pd.DataFrame(results_data)
        # output_filename = f"langdetect_results_{language_name.lower()}.csv" # For Reddit Datasets
        output_filename = f"langdetect_news_results_{language_name.lower()}.csv"
        results_df.to_csv(output_filename, index=False, encoding='utf-8')
        
        print(f"\\nResults saved to: {output_filename}")
        print(f"Total posts analyzed: {len(results_data)}")
        
        # Show summary statistics
        print(f"\\n{'='*40}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*40}")
        
        # Language distribution
        language_counts = results_df['predicted_language'].value_counts()
        print(f"\\nDetected languages:")
        for lang, count in language_counts.items():
            percentage = (count / len(results_data)) * 100
            print(f"  {lang}: {count} posts ({percentage:.1f}%)")
        
        # Check for English detection specifically
        english_count = language_counts.get('en', 0)
        if english_count > 0:
            english_percentage = (english_count / len(results_data)) * 100
            print(f"\\nEnglish content detected: {english_count} posts ({english_percentage:.1f}%)")
        
        return results_df
    else:
        print("No valid content found to analyze")
        return None

def main():
    """Main function to analyze all three datasets"""
    # Analyze each dataset
    # yoruba_results = analyze_languages_with_langdetect('yoruba_reddit_posts.csv', 'Yoruba')
    # kinyarwanda_results = analyze_languages_with_langdetect('kinyarwanda_reddit_posts.csv', 'Kinyarwanda')
    # amharic_results = analyze_languages_with_langdetect('amharic_reddit_posts.csv', 'Amharic')

    # DETECTING NEWS DATA
    yoruba_results = analyze_languages_with_langdetect('yoruba_news.csv', 'Yoruba')
    kinyarwanda_results = analyze_languages_with_langdetect('kinyarwanda.csv', 'Kinyarwanda')
    amharic_results = analyze_languages_with_langdetect('amharic_csv.csv', 'Amharic')
    
    print("\\n" + "="*60)
    print("ALL LANGDETECT ANALYSES COMPLETED!")
    print("="*60)
    print("Check the generated CSV files for detailed results.")
    print("\\nKey differences from AfroLID:")
    print("- langdetect supports English (en) detection")
    print("- langdetect supports 55 languages vs AfroLID's 518 African languages")
    print("- langdetect is deterministic (no confidence scores)")

if __name__ == "__main__":
    main()
