#!/usr/bin/env python3
"""
Vietnamese Text-to-Phonemes Converter (Southern Dialect)
Specialized for Southern Vietnamese with easy customization options.
"""

import re
import unicodedata

# Vietnamese vowel mapping to IPA (Southern dialect)
VOWEL_MAP = {
    'a': 'a', 'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'ă', 'ằ': 'ă', 'ắ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
    'â': 'â', 'ầ': 'â', 'ấ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
    'e': 'ɛ', 'è': 'ɛ', 'é': 'ɛ', 'ẻ': 'ɛ', 'ẽ': 'ɛ', 'ẹ': 'ɛ',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'i': 'i', 'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'o': 'ɔ', 'ò': 'ɔ', 'ó': 'ɔ', 'ỏ': 'ɔ', 'õ': 'ɔ', 'ọ': 'ɔ',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'ɤ', 'ờ': 'ɤ', 'ớ': 'ɤ', 'ở': 'ɤ', 'ỡ': 'ɤ', 'ợ': 'ɤ',
    'u': 'u', 'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'ɯ', 'ừ': 'ɯ', 'ứ': 'ɯ', 'ử': 'ɯ', 'ữ': 'ɯ', 'ự': 'ɯ',
    'y': 'i', 'ỳ': 'i', 'ý': 'i', 'ỷ': 'i', 'ỹ': 'i', 'ỵ': 'i',
}

# Tone marks mapping (Southern pronunciation)
TONE_MAP = {
    '': '',           # ngang (level)
    'a': '', 'ă': '', 'â': '', 'e': '', 'ê': '', 'i': '', 'o': '', 'ô': '', 'ơ': '', 'u': '', 'ư': '', 'y': '',
    'á': '˥', 'ắ': '˥', 'ấ': '˥', 'é': '˥', 'ế': '˥', 'í': '˥', 'ó': '˥', 'ố': '˥', 'ớ': '˥', 'ú': '˥', 'ứ': '˥', 'ý': '˥',
    'à': '˨˩', 'ằ': '˨˩', 'ầ': '˨˩', 'è': '˨˩', 'ề': '˨˩', 'ì': '˨˩', 'ò': '˨˩', 'ồ': '˨˩', 'ờ': '˨˩', 'ù': '˨˩', 'ừ': '˨˩', 'ỳ': '˨˩',
    'ả': '˧˩˧', 'ẳ': '˧˩˧', 'ẩ': '˧˩˧', 'ẻ': '˧˩˧', 'ể': '˧˩˧', 'ỉ': '˧˩˧', 'ỏ': '˧˩˧', 'ổ': '˧˩˧', 'ở': '˧˩˧', 'ủ': '˧˩˧', 'ử': '˧˩˧', 'ỷ': '˧˩˧',
    'ã': '˧˥', 'ẵ': '˧˥', 'ẫ': '˧˥', 'ẽ': '˧˥', 'ễ': '˧˥', 'ĩ': '˧˥', 'õ': '˧˥', 'ỗ': '˧˥', 'ỡ': '˧˥', 'ũ': '˧˥', 'ữ': '˧˥', 'ỹ': '˧˥',
    'ạ': '˨', 'ặ': '˨', 'ậ': '˨', 'ẹ': '˨', 'ệ': '˨', 'ị': '˨', 'ọ': '˨', 'ộ': '˨', 'ợ': '˨', 'ụ': '˨', 'ự': '˨', 'ỵ': '˨',
}

def get_base_vowel(char):
    """Get the base vowel without tone mark."""
    return VOWEL_MAP.get(char, char)

def get_tone(char):
    """Get the tone mark for a character."""
    return TONE_MAP.get(char, '')

def normalize_text(text):
    """Normalize Vietnamese text to NFC form."""
    return unicodedata.normalize('NFC', text)

def parse_syllable(syllable, config=None):
    """
    Parse a Vietnamese syllable into phonemes.
    
    A Vietnamese syllable structure: (initial consonant) + (medial) + (nucleus vowel) + (final consonant) + (tone)
    """
    if config is None:
        config = {}
    
    # Default configuration for Southern dialect
    default_config = {
        'r_as_g': True,      # r -> /ɣ/ (Southern default: r sounds like g)
        'd_as_z': True,      # d -> /z/ (Southern)
        'v_as_j': True,      # v -> /j/ (Southern)
        'x_as_s': True,      # x -> /s/
        'gi_as_z': True,     # gi -> /z/ (Southern)
        'ch_as_c': True,     # ch -> /c/ 
        'tr_as_ch': False,   # tr -> /ʈʂ/ (Northern) or /tʂ/; Southern often merges with ch
        'ng_final': 'ŋ',     # ng/nh finals
        'nh_final': 'ɲ',
    }
    config = {**default_config, **config}
    
    syllable = syllable.strip().lower()
    if not syllable:
        return []
    
    phonemes = []
    
    # Handle special initial clusters
    initials = {
        'gi': 'z' if config['gi_as_z'] else 'z',
        'ngh': 'ŋ',
        'ng': 'ŋ',
        'nh': 'ɲ',
        'gh': 'ɣ',
        'kh': 'x',
        'th': 'tʰ',
        'ph': 'f',
        'tr': 'ʈʂ' if config['tr_as_ch'] else 'tʂ',
        'ch': 'c' if config['ch_as_c'] else 'c',
    }
    
    # Single consonants
    single_initials = {
        'b': 'b',
        'c': 'k',
        'd': 'z' if config['d_as_z'] else 'z',
        'đ': 'ɗ',
        'g': 'ɣ',
        'h': 'h',
        'k': 'k',
        'l': 'l',
        'm': 'm',
        'n': 'n',
        'p': 'p',
        'q': 'k',  # q is always followed by u, pronounced as kw or k
        'r': 'ɣ' if config['r_as_g'] else 'ɹ',  # CUSTOMIZABLE: r as g or English r
        's': 'ʂ' if not config.get('x_as_s', True) else 's',
        't': 't',
        'v': 'v' if not config['v_as_j'] else 'j',
        'x': 's' if config['x_as_s'] else 's',
    }
    
    # Try to match initial consonant(s)
    initial = ''
    remaining = syllable
    
    # Check for two-character initials first
    for init_two in ['gi', 'ngh', 'ng', 'nh', 'gh', 'kh', 'th', 'ph', 'tr', 'ch', 'qu']:
        if syllable.startswith(init_two):
            # Special case: qu is /k/ + /w/
            if init_two == 'qu':
                phonemes.append('k')
                phonemes.append('w')
                remaining = syllable[2:]
            else:
                phonemes.append(initials.get(init_two, init_two))
                remaining = syllable[len(init_two):]
            initial = init_two
            break
    
    # If no two-char initial found, check single char
    if not initial and syllable:
        first_char = syllable[0]
        if first_char in single_initials:
            phonemes.append(single_initials[first_char])
            remaining = syllable[1:]
        elif first_char in VOWEL_MAP:
            # Syllable starts with vowel
            remaining = syllable
        else:
            # Unknown character, skip
            remaining = syllable[1:] if len(syllable) > 1 else ''
    
    # Now process the vowel nucleus and final
    if remaining:
        # Find the main vowel(s)
        vowels_in_syllable = []
        vowel_indices = []
        
        for i, char in enumerate(remaining):
            if char in VOWEL_MAP:
                vowels_in_syllable.append(char)
                vowel_indices.append(i)
        
        # Process vowels and any medial/final
        if vowels_in_syllable:
            # Handle diphthongs/triphthongs
            vowel_str = ''.join(vowels_in_syllable)
            
            # Common vowel combinations in Vietnamese
            vowel_combos = {
                'ia': 'iə', 'iê': 'iə', 'yê': 'iə', 'ya': 'iə',
                'ua': 'uə', 'uô': 'uə',
                'ưa': 'ɯə', 'ươ': 'ɯə',
                'ai': 'aj', 'ay': 'aj',
                'ao': 'aw',
                'au': 'aw',
                'âu': 'əw',
                'ei': 'ej',
                'êu': 'ew',
                'oi': 'ɔj',
                'ôi': 'oj',
                'ơi': 'ɤj',
                'ui': 'uj',
                'uy': 'uj',
                'ưu': 'ɯw',
                'eo': 'ɛw',
                'ieu': 'iəw', 'yêu': 'iəw',
                'uoi': 'uɔj',
                'ươi': 'ɯəj',
                'oai': 'waːj',
                'oay': 'waːj',
                'oeo': 'ɛw',
                'uây': 'wəj',
                'uyê': 'wiə',
                'uyên': 'wiən',
            }
            
            # Check if we have a known combination
            matched_combo = False
            for combo_len in [4, 3, 2]:  # Try longest first
                if len(vowel_str) >= combo_len:
                    test_combo = vowel_str[:combo_len]
                    if test_combo in vowel_combos:
                        # Add the combined vowel sound
                        combo_phoneme = vowel_combos[test_combo]
                        # Get tone from the last vowel in the combo
                        last_vowel = vowels_in_syllable[combo_len - 1]
                        tone = get_tone(last_vowel)
                        phonemes.append(combo_phoneme)
                        if tone:
                            phonemes.append(tone)
                        
                        # Remove processed vowels from remaining
                        # Find position after the last vowel of the combo
                        last_vowel_idx = vowel_indices[combo_len - 1]
                        remaining = remaining[last_vowel_idx + 1:]
                        matched_combo = True
                        break
            
            if not matched_combo:
                # Process vowels individually
                for i, v in enumerate(vowels_in_syllable):
                    base_v = get_base_vowel(v)
                    tone = get_tone(v)
                    phonemes.append(VOWEL_MAP[v])
                    # Only add tone once, on the main vowel (usually the last one in the nucleus)
                    if i == len(vowels_in_syllable) - 1 and tone:
                        phonemes.append(tone)
                
                # Remove all vowels from remaining
                new_remaining = ''
                for i, char in enumerate(remaining):
                    if char not in VOWEL_MAP:
                        new_remaining += char
                remaining = new_remaining
        
        # Process final consonants
        if remaining:
            finals = {
                'c': 'k̚',  # Unreleased stop
                'ch': 'c̟̚',
                'm': 'm',
                'n': 'n',
                'ng': 'ŋ',
                'nh': 'ɲ',
                'p': 'p̚',  # Unreleased stop
                't': 't̚',  # Unreleased stop
                'u': 'w',  # Final u/o becomes /w/
                'o': 'w',
                'i': 'j',  # Final i/y becomes /j/
                'y': 'j',
            }
            
            # Check for two-char finals
            if len(remaining) >= 2 and remaining[-2:] in ['ch', 'ng', 'nh']:
                final = remaining[-2:]
                phonemes.append(finals.get(final, final))
                remaining = remaining[:-2]
            elif remaining:
                # Process remaining characters as finals
                for char in remaining:
                    if char in finals:
                        phonemes.append(finals[char])
                    elif char in VOWEL_MAP:
                        # Final vowel becomes glide
                        if char in ['u', 'o']:
                            phonemes.append('w')
                        elif char in ['i', 'y']:
                            phonemes.append('j')
    
    return phonemes

def text_to_phonemes(text, config=None):
    """
    Convert Vietnamese text to phonemes using Southern dialect rules.
    
    Args:
        text: Input Vietnamese text
        config: Dictionary with customization options
        
    Returns:
        String of space-separated phonemes
    """
    if config is None:
        config = {}
    
    text = normalize_text(text)
    
    # Split into words/syllables
    # Vietnamese is written with spaces between syllables
    syllables = text.split()
    
    all_phonemes = []
    for syllable in syllables:
        # Remove punctuation for processing
        clean_syllable = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', '', syllable)
        punctuation = re.sub(r'[\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', '', syllable)
        
        if clean_syllable:
            syllable_phonemes = parse_syllable(clean_syllable, config)
            all_phonemes.extend(syllable_phonemes)
        
        # Add back punctuation as separate tokens
        if punctuation:
            all_phonemes.append(punctuation)
    
    return ' '.join(all_phonemes)

def text_to_phonemes_compact(text, config=None):
    """
    Convert Vietnamese text to phonemes without spaces (for Piper CSV format).
    
    Args:
        text: Input Vietnamese text
        config: Dictionary with customization options
        
    Returns:
        String of phonemes without spaces
    """
    phonemes = text_to_phonemes(text, config)
    # Remove spaces but keep phonemes together
    return phonemes.replace(' ', '')

def create_piper_csv_line(vietnamese_text, audio_filename, config=None):
    """
    Create a line suitable for Piper training CSV with phonemes.
    
    Args:
        vietnamese_text: Original Vietnamese text
        audio_filename: Path to corresponding audio file
        config: Configuration for phoneme conversion
        
    Returns:
        String in format: filename|phonemes
    """
    phonemes = text_to_phonemes_compact(vietnamese_text, config)
    return f"{audio_filename}|{phonemes}"


if __name__ == "__main__":
    print("Vietnamese Text-to-Phonemes Converter (Southern Dialect)")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        "Xin chào",
        "Rổ rá",
        "Con cá rô",
        "Chào anh",
        "Tôi yêu Việt Nam",
        "Cơm ngon",
    ]
    
    print("\n--- Default Southern Dialect (r as /ɣ/) ---")
    for test in test_cases:
        print(f"Input: {test}")
        print(f"Output: {text_to_phonemes(test)}")
        print()
    
    print("\n--- With r as English /ɹ/ ---")
    custom_config = {'r_as_g': False}
    for test in ["Rổ rá", "Con cá rô"]:
        print(f"Input: {test}")
        print(f"Output: {text_to_phonemes(test, config=custom_config)}")
        print()
    
    print("\n--- Piper CSV Format Examples ---")
    print(create_piper_csv_line("Rổ rá", "ro_ra.wav", {'r_as_g': False}))
    print(create_piper_csv_line("Xin chào", "xin_chao.wav"))
    print(create_piper_csv_line("Con cá rô", "con_ca_ro.wav", {'r_as_g': False}))
