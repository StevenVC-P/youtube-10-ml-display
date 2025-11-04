"""Remove all emoji characters from Python files to fix Windows console encoding issues."""
import os
import re
from pathlib import Path

# Emoji to ASCII replacements
EMOJI_REPLACEMENTS = {
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '📁': '[FILE]',
    '📊': '[STATS]',
    '🎬': '[VIDEO]',
    '📹': '[VIDEO]',
    '🧠': '[ML]',
    '📍': '[INFO]',
    '🔍': '[DEBUG]',
    '💡': '[TIP]',
    '🔧': '[FIX]',
}

def remove_emojis_from_file(file_path):
    """Remove emoji characters from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace known emojis
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)
        
        # Remove any remaining non-ASCII characters (except common ones like ×, ε)
        # Keep mathematical symbols that are commonly used
        keep_chars = {'×', 'ε', '@'}
        
        def replace_char(match):
            char = match.group(0)
            if char in keep_chars:
                return char
            # Check if it's in a string literal
            return '[?]'
        
        # Only replace emojis, not all non-ASCII
        # This regex matches emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        content = emoji_pattern.sub('[?]', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    """Remove emojis from all Python files in training directory."""
    training_dir = Path('training')
    fixed_count = 0
    
    for py_file in training_dir.glob('**/*.py'):
        if remove_emojis_from_file(py_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()

