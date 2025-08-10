import pandas as pd
import numpy as np
from transformers import pipeline
import re
import warnings

warnings.filterwarnings('ignore')

def clean_text(text):
    """Clean text by removing URLs, special characters, and extra whitespace"""
    if pd.isna(text) or text == '':
        return ''
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', str(text))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', str(text)).strip()
    
    return text

def analyze_languages_with_afrolid(csv_file, language_name):
    """Analyze CSV file using AfroLID for language detection"""
    print(f"\n{'='*60}")
    print(f"AFROLID ANALYSIS: {language_name.upper()}")
    print(f"{'='*60}")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} posts from {csv_file}")
    
    # Initialize AfroLID
    print("Initializing AfroLID...")
    afrolid = pipeline("text-classification", model='UBC-NLP/afrolid_1.5')
    print("AfroLID initialized successfully!")
    
    # Prepare results data
    results_data = []
    
    # Process each post
    for idx, row in df.iterrows():
        # content = row['content'] if pd.notna(row['content']) else '' # For the Reddit Dataset
        content = row['Text'] if pd.notna(row['Text']) else ''
        
        if content and len(content.strip()) > 10:  # Only process non-empty content
            # Clean the text
            cleaned_content = clean_text(content)
            
            if cleaned_content:
                try:
                    # Get AfroLID prediction
                    result = afrolid(cleaned_content)
                    prediction = result[0] if result else None
                    
                    if prediction:
                        results_data.append({
                            'content': cleaned_content,
                            'predicted_language': prediction['label'],
                            'confidence_score': prediction['score']
                        })
                        
                        # Print progress
                        print(f"Post {idx+1}: {cleaned_content[:100]}...")
                        print(f"  → Predicted: {prediction['label']} (confidence: {prediction['score']:.3f})")
                        print()
                    
                except Exception as e:
                    print(f"Error processing post {idx+1}: {e}")
                    continue
    
    # Save results to CSV
    if results_data:
        results_df = pd.DataFrame(results_data)
        # output_filename = f"afrolid_results_{language_name.lower()}.csv" # For Reddit Datasets
        output_filename = f"afrolid_news_results_{language_name.lower()}.csv"
        results_df.to_csv(output_filename, index=False, encoding='utf-8')
        
        print(f"\nResults saved to: {output_filename}")
        print(f"Total posts analyzed: {len(results_data)}")
        
        # Show summary statistics
        print(f"\n{'='*40}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*40}")
        
        # Language distribution
        language_counts = results_df['predicted_language'].value_counts()
        print(f"\nDetected languages:")
        for lang, count in language_counts.items():
            percentage = (count / len(results_data)) * 100
            print(f"  {lang}: {count} posts ({percentage:.1f}%)")
        
        # Confidence statistics
        avg_confidence = results_df['confidence_score'].mean()
        print(f"\nAverage confidence score: {avg_confidence:.3f}")
        
        return results_df
    else:
        print("No valid content found to analyze")
        return None

def main():
    """Main function to analyze all three datasets"""
    # Analyze each dataset
    # yoruba_results = analyze_languages_with_afrolid('yoruba_reddit_posts.csv', 'Yoruba')
    # kinyarwanda_results = analyze_languages_with_afrolid('kinyarwanda_reddit_posts.csv', 'Kinyarwanda')
    # amharic_results = analyze_languages_with_afrolid('amharic_reddit_posts.csv', 'Amharic')

    # DETECTING NEWS DATA
    yoruba_results = analyze_languages_with_afrolid('yoruba_news.csv', 'Yoruba')
    kinyarwanda_results = analyze_languages_with_afrolid('kinyarwanda.csv', 'Kinyarwanda')
    amharic_results = analyze_languages_with_afrolid('amharic_csv.csv', 'Amharic')
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETED!")
    print("="*60)
    print("Check the generated CSV files for detailed results.")

if __name__ == "__main__":
    main()
