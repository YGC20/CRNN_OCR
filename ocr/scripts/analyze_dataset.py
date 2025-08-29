
import os
from collections import Counter
import codecs

def analyze_character_distribution(labels_dir):
    """
    Analyzes the character distribution in the dataset.

    Args:
        labels_dir (str): Path to the directory containing label files.
    """
    char_counter = Counter()
    
    print(f"Analyzing labels in: {labels_dir}")

    if not os.path.isdir(labels_dir):
        print(f"Error: Directory not found at {labels_dir}")
        return

    label_files = os.listdir(labels_dir)
    total_files = len(label_files)
    print(f"Found {total_files} label files.")

    for i, filename in enumerate(label_files):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_dir, filename)
            try:
                # Use codecs.open for robust encoding handling
                with codecs.open(file_path, 'r', encoding='utf-8') as f:
                    label = f.read().strip()
                    char_counter.update(label)
            except Exception as e:
                print(f"Could not read file {filename}: {e}")
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{total_files} files...")

    print("\n--- Character Distribution Analysis ---")
    if not char_counter:
        print("No characters found or counted.")
        return

    # Sort by frequency (most common first)
    sorted_chars = char_counter.most_common()

    print(f"Total unique characters: {len(sorted_chars)}\n")

    print("Top 30 most common characters:")
    for char, count in sorted_chars[:30]:
        print(f"  '{char}': {count}")

    print("\nTop 30 least common characters:")
    for char, count in sorted_chars[-30:]:
        print(f"  '{char}': {count}")


if __name__ == '__main__':
    # Path to the labels directory for the full virtual dataset
    virtual_data_labels_dir = os.path.join('ocr', 'data', 'virtual_data', 'labels')
    analyze_character_distribution(virtual_data_labels_dir)
