"""
Vietnamese Text to Phonemes Converter
Specialized for Southern Vietnamese dialect

This module provides functions to convert Vietnamese text to phonemes
for use with Piper TTS training (Custom Phonemes mode).

Usage:
    from vietnamese_phonemizer import text_to_phonemes
    
    # Basic usage
    phonemes = text_to_phonemes("Xin chào Việt Nam")
    
    # Customize pronunciation (e.g., make 'r' sound like English 'r' instead of 'g')
    config = {
        'r_as_g': False,  # Set to True for traditional Southern accent (r→g)
        'd_as_z': True,   # Southern: d→z (gi→z)
        'v_as_j': True,   # Southern: v→j (v→ʒ-like)
    }
    phonemes = text_to_phonemes("Rổ rá", config=config)
"""

import re
import unicodedata
from typing import Dict, Optional, List


# Default Southern Vietnamese phoneme mappings
# These can be customized via the config parameter
DEFAULT_CONFIG = {
    # Consonant mappings
    'r_as_g': False,      # If True: r → g (traditional Southern), False: r → ɹ (English-like)
    'd_as_z': True,       # Southern: d → z (as in 'dạ' → zaː˨˩)
    'gi_as_z': True,      # Southern: gi → z (as in 'giường' → zɨəŋ˧˨)
    'v_as_j': True,       # Southern: v → j (as in 'vui' → juj˧)
    'x_as_s': True,       # Southern: x → s (as in 'xa' → saː˧)
    'ch_as_c': False,     # If True: ch → c (some Southern speakers)
    'nh_as_ɲ': True,      # Standard: nh → ɲ
    'ng_as_ŋ': True,      # Standard: ng → ŋ
    'ngh_as_ŋ': True,     # Standard: ngh → ŋ
    'kh_as_x': True,      # Standard: kh → x
    'th_as_t̪ʰ': True,     # Aspirated t
    'ph_as_f': True,      # Southern often: ph → f
    
    # Vowel mappings (Southern specific)
    'a_after_i': 'ɛ',     # ia → iɛ (Southern: prefer open vowel)
    'a_after_u': 'ɔ',     # ua → uɔ
    'o_after_e': 'ɔ',     # eo → ɛɔ
    'u_after_o': 'w',     # ou → ow
    
    # Tone markers (6 tones in Vietnamese)
    # Level (ngang): no mark - ˧ (mid level)
    # Acute (sắc): ́ - ˧˥ (rising)
    # Grave (huyền): ̀ - ˨˩ (low falling)
    # Hook (hỏi): ̉ - ˧˩˧ (dipping-rising)
    # Tilde (ngã): ̃ - ˧˦ˀ˥ (creaky rising)
    # Dot (nặng): ̣ - ˨˩ˀ˥ (low falling constricted)
    'tone_marks': {
        'ngang': '˧',
        'sac': '˧˥',
        'huyen': '˨˩',
        'hoi': '˧˩˧',
        'nga': '˧˦ˀ˥',
        'nang': '˨˩ˀ˥',
    },
    
    # Final consonants
    'c_final': 'k',
    't_final': 't',
    'p_final': 'p',
    'ch_final': 'c',  # Southern: ch final → c
    'nh_final': 'ɲ',
    'ng_final': 'ŋ',
    'n_final': 'n',
    'm_final': 'm',
}


# IPA symbols for Vietnamese consonants (initial position)
INITIAL_CONSONANTS = {
    'b': 'b',
    'c': 'k',
    'd': 'z',      # Southern: d → z
    'đ': 'ɗ',
    'g': 'ɣ',
    'gh': 'ɣ',
    'h': 'h',
    'k': 'k',
    'kh': 'x',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ng': 'ŋ',
    'ngh': 'ŋ',
    'nh': 'ɲ',
    'p': 'p',
    'ph': 'f',
    'q': 'k',
    'r': 'ɣ',      # Traditional Southern: r → ɣ/g
    's': 'ʂ',
    't': 't',
    'th': 't̪ʰ',
    'tr': 'ʈ',
    'v': 'v',
    'x': 's',
}

# Final consonants
FINAL_CONSONANTS = {
    'c': 'k',
    'ch': 'c',
    'm': 'm',
    'n': 'n',
    'ng': 'ŋ',
    'nh': 'ɲ',
    'p': 'p',
    't': 't',
    'ch': 'c',
}

# Vowels and diphthongs
VOWELS = {
    'a': 'aː',
    'ă': 'a',
    'â': 'ɜ',
    'e': 'ɛ',
    'ê': 'e',
    'i': 'i',
    'y': 'i',
    'o': 'ɔ',
    'ô': 'o',
    'ơ': 'ɤ',
    'u': 'u',
    'ư': 'ɯ',
}

# Diphthongs/Triphthongs
DIPHTHONGS = {
    'ia': 'iɛ',
    'yê': 'iɛ',
    'ưa': 'ɯɤ',
    'ua': 'uɔ',
    'ai': 'aːj',
    'ay': 'aj',
    'ây': 'ɜj',
    'ao': 'aːw',
    'au': 'aw',
    'âu': 'ɜw',
    'ei': 'ej',
    'êu': 'ew',
    'oi': 'ɔj',
    'ôi': 'oj',
    'ơi': 'ɤj',
    'ui': 'uj',
    'ưi': 'ɯj',
    'uo': 'uɔ',
    'ươ': 'ɯɤ',
    'iêu': 'iɛw',
    'yêu': 'iɛw',
    'ươi': 'ɯɤj',
    'uôi': 'uɔj',
    'uyê': 'wiɛ',
    'uyên': 'wiɛn',
    'oai': 'waːj',
    'oay': 'waj',
    'oe': 'wɛ',
    'oeo': 'wɛɔ',
    'ue': 'wɛ',
    'uê': 'we',
    'uôi': 'uɔj',
}


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text (handle encoding variations)."""
    # Normalize Unicode (NFC for consistent representation)
    text = unicodedata.normalize('NFC', text)
    return text


def extract_tone(char: str) -> tuple:
    """Extract tone mark from a character and return (base_char, tone_name)."""
    tone_map = {
        '\u0301': 'sac',      # ́ acute
        '\u0300': 'huyen',    # ̀ grave
        '\u0309': 'hoi',      # ̉ hook
        '\u0303': 'nga',      # ̃ tilde
        '\u0323': 'nang',     # ̣ dot below
    }
    
    # Decompose the character to separate tone marks
    decomposed = unicodedata.normalize('NFD', char)
    
    base_char = ''
    tone = 'ngang'  # default: level tone
    
    for c in decomposed:
        if c in tone_map:
            tone = tone_map[c]
        elif c not in tone_map.values():
            base_char += c
    
    return base_char, tone


def get_initial_consonant(syllable: str, config: Dict) -> tuple:
    """Extract initial consonant from syllable and return (consonant_ipa, remainder)."""
    # Sort by length (longest first) to match multi-char consonants first
    initials = sorted(
        [k for k in INITIAL_CONSONANTS.keys() if syllable.startswith(k)],
        key=len,
        reverse=True
    )
    
    if not initials:
        return '', syllable
    
    initial = initials[0]
    remainder = syllable[len(initial):]
    
    # Apply custom mappings from config
    ipa = INITIAL_CONSONANTS[initial]
    
    # Special handling based on config
    if initial == 'r':
        if config.get('r_as_g', False):
            ipa = 'ɣ'  # Traditional Southern
        else:
            ipa = 'ɹ'  # English-like r
    
    if initial == 'd':
        ipa = 'z' if config.get('d_as_z', True) else 'z'
    
    if initial == 'gi':
        ipa = 'z' if config.get('gi_as_z', True) else 'z'
    
    if initial == 'v':
        ipa = 'j' if config.get('v_as_j', True) else 'v'
    
    if initial == 'x':
        ipa = 's' if config.get('x_as_s', True) else 'ʂ'
    
    if initial == 'ph':
        ipa = 'f' if config.get('ph_as_f', True) else 'f'
    
    return ipa, remainder


def get_final_consonant(syllable: str, config: Dict) -> tuple:
    """Extract final consonant from syllable and return (remainder, final_ipa)."""
    finals = sorted(
        [k for k in FINAL_CONSONANTS.keys() if syllable.endswith(k)],
        key=len,
        reverse=True
    )
    
    if not finals:
        return syllable, ''
    
    final = finals[0]
    remainder = syllable[:-len(final)]
    ipa = FINAL_CONSONANTS[final]
    
    return remainder, ipa


def process_vowels(vowel_part: str, config: Dict) -> str:
    """Process vowels and diphthongs."""
    # Try to match diphthongs/triphthongs first (longest match)
    for dip in sorted(DIPHTHONGS.keys(), key=len, reverse=True):
        if dip in vowel_part:
            return vowel_part.replace(dip, DIPHTHONGS[dip])
    
    # Process individual vowels
    result = []
    i = 0
    while i < len(vowel_part):
        matched = False
        # Try multi-char vowels first
        for length in [2, 1]:
            if i + length <= len(vowel_part):
                chunk = vowel_part[i:i+length]
                if chunk in VOWELS:
                    base, tone = extract_tone(chunk[-1]) if any(ord(c) > 127 for c in chunk) else (chunk[-1], 'ngang')
                    if length == 1:
                        result.append(VOWELS.get(chunk, chunk))
                    else:
                        result.append(chunk)
                    matched = True
                    i += length
                    break
        if not matched:
            result.append(vowel_part[i])
            i += 1
    
    return ''.join(result)


def phonemize_syllable(syllable: str, config: Dict) -> str:
    """Convert a single Vietnamese syllable to IPA phonemes."""
    if not syllable:
        return ''
    
    syllable = syllable.lower().strip()
    
    # Extract initial consonant
    initial_ipa, remainder = get_initial_consonant(syllable, config)
    
    # Extract final consonant
    vowel_part, final_ipa = get_final_consonant(remainder, config)
    
    # Process vowels (with tone handling)
    vowel_ipa = ''
    tone = 'ngang'
    
    # Extract tone from vowel part
    processed_vowels = []
    for char in vowel_part:
        base, char_tone = extract_tone(char)
        if char_tone != 'ngang':
            tone = char_tone
        if base in VOWELS:
            processed_vowels.append(VOWELS[base])
        else:
            processed_vowels.append(base)
    
    vowel_ipa = ''.join(processed_vowels)
    
    # Build phoneme string
    phonemes = []
    if initial_ipa:
        phonemes.append(initial_ipa)
    if vowel_ipa:
        phonemes.append(vowel_ipa)
    if final_ipa:
        phonemes.append(final_ipa)
    
    # Add tone marker at the end
    tone_marker = config['tone_marks'].get(tone, '')
    if tone_marker:
        phonemes.append(tone_marker)
    
    return ' '.join(phonemes)


def text_to_phonemes(text: str, config: Optional[Dict] = None) -> str:
    """
    Convert Vietnamese text to phonemes (IPA format).
    
    Args:
        text: Vietnamese text string
        config: Optional configuration dictionary for customizing pronunciation
        
    Returns:
        String of phonemes separated by spaces, suitable for Piper TTS training
    
    Example:
        >>> text_to_phonemes("Xin chào")
        's ɨ n ˧ c aː w ˨˩'
        
        >>> # Make 'r' sound like English 'r' instead of Southern 'g'
        >>> config = {'r_as_g': False}
        >>> text_to_phonemes("Rổ rá", config=config)
        'ɹ o ˧˦ˀ˥ ɹ a ˧˥'
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    # Normalize input text
    text = normalize_text(text)
    
    # Split into words/syllables (Vietnamese is written with spaces between syllables)
    syllables = text.split()
    
    phonemes_list = []
    for syllable in syllables:
        # Handle punctuation
        if not any(c.isalpha() for c in syllable):
            phonemes_list.append(syllable)
            continue
        
        syllable_phonemes = phonemize_syllable(syllable, config)
        if syllable_phonemes:
            phonemes_list.append(syllable_phonemes)
    
    return ' '.join(phonemes_list)


def text_to_phonemes_piper(text: str, config: Optional[Dict] = None) -> str:
    """
    Convert Vietnamese text to phonemes in Piper-compatible format.
    
    This returns phonemes as a continuous string of UTF-8 codepoints
    (no spaces between phonemes within a syllable), which is the format
    expected by Piper's Custom Phonemes mode.
    
    Args:
        text: Vietnamese text string
        config: Optional configuration dictionary
        
    Returns:
        String of phonemes suitable for Piper CSV (phoneme_type=text mode)
    
    Example:
        >>> text_to_phonemes_piper("Xin chào")
        'sɨn˧ caːw˨˩'
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    text = normalize_text(text)
    syllables = text.split()
    
    phonemes_list = []
    for syllable in syllables:
        if not any(c.isalpha() for c in syllable):
            phonemes_list.append(syllable)
            continue
        
        syllable_phonemes = phonemize_syllable(syllable, config)
        # Remove spaces within syllable for Piper format
        syllable_phonemes = syllable_phonemes.replace(' ', '')
        if syllable_phonemes:
            phonemes_list.append(syllable_phonemes)
    
    return ' '.join(phonemes_list)


# Convenience function for creating Piper training CSV
def create_piper_csv_line(audio_file: str, text: str, config: Optional[Dict] = None) -> str:
    """
    Create a line for Piper training CSV with custom phonemes.
    
    Format: audio_file.wav|phonemes
    
    Args:
        audio_file: Path to audio file (or just filename)
        text: Vietnamese text to phonemize
        config: Optional configuration dictionary
        
    Returns:
        CSV line in format: filename.wav|phonemes
    """
    phonemes = text_to_phonemes_piper(text, config)
    return f"{audio_file}|{phonemes}"


if __name__ == "__main__":
    # Demo usage
    print("Vietnamese to Phonemes Converter (Southern Dialect)")
    print("=" * 50)
    
    test_sentences = [
        "Xin chào Việt Nam",
        "Rổ rá",
        "Con cá",
        "Chào anh",
        "Tôi yêu Việt Nam",
        "Cơm chưa",
    ]
    
    print("\nDefault Southern accent (r → ɣ):")
    for sentence in test_sentences:
        phonemes = text_to_phonemes(sentence)
        piper_format = text_to_phonemes_piper(sentence)
        print(f"  {sentence}")
        print(f"    → {phonemes}")
        print(f"    → Piper: {piper_format}")
        print()
    
    print("\nWith English-like 'r' (r → ɹ):")
    config = {'r_as_g': False}
    for sentence in test_sentences:
        if 'r' in sentence.lower() or 'R' in sentence:
            phonemes = text_to_phonemes(sentence, config)
            piper_format = text_to_phonemes_piper(sentence, config)
            print(f"  {sentence}")
            print(f"    → {phonemes}")
            print(f"    → Piper: {piper_format}")
            print()
    
    print("\nExample CSV lines for Piper training:")
    print(create_piper_csv_line("utt1.wav", "Xin chào"))
    print(create_piper_csv_line("utt2.wav", "Rổ rá", {'r_as_g': False}))
