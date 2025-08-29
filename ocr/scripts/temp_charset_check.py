
CHOSUNG_LIST = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JOONGSUNG_LIST = ['ㅛ', 'ㅕ', 'ㅑ', 'ㅐ', 'ㅔ', 'ㅗ', 'ㅓ', 'ㅏ', 'ㅣ', 'ㅠ', 'ㅜ', 'ㅡ']

def combine_hangul(chosung, joongsung):
    CHOSUNG_MAP = {'ㄱ': 0, 'ㄲ': 1, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 4, 'ㄹ': 5, 'ㅁ': 6, 'ㅂ': 7, 'ㅃ': 8, 'ㅅ': 9, 'ㅆ': 10, 'ㅇ': 11, 'ㅈ': 12, 'ㅉ': 13, 'ㅊ': 14, 'ㅋ': 15, 'ㅌ': 16, 'ㅍ': 17, 'ㅎ': 18}
    JOONGSUNG_MAP = {'ㅏ': 0, 'ㅐ': 1, 'ㅑ': 2, 'ㅒ': 3, 'ㅓ': 4, 'ㅔ': 5, 'ㅕ': 6, 'ㅖ': 7, 'ㅗ': 8, 'ㅘ': 9, 'ㅫ': 10, 'ㅚ': 11, 'ㅛ': 12, 'ㅜ': 13, 'ㅝ': 14, 'ㅞ': 15, 'ㅟ': 16, 'ㅠ': 17, 'ㅡ': 18, 'ㅢ': 19, 'ㅣ': 20}
    chosung_idx = CHOSUNG_MAP.get(chosung)
    joongsung_idx = JOONGSUNG_MAP.get(joongsung)
    if chosung_idx is None or joongsung_idx is None: return None
    return chr(0xAC00 + chosung_idx * 21 * 28 + joongsung_idx * 28)

combined_hangul = [combine_hangul(c, j) for c in CHOSUNG_LIST for j in JOONGSUNG_LIST if combine_hangul(c, j) is not None]
JAMO_CHARS = CHOSUNG_LIST + JOONGSUNG_LIST
ENGLISH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUMBER_CHARS = "0123456789"
VIRTUAL_CHARSET = "".join(combined_hangul) + "".join(JAMO_CHARS) + ENGLISH_CHARS + NUMBER_CHARS

print(f"VIRTUAL_CHARSET length: {len(VIRTUAL_CHARSET)}")
print(f"VIRTUAL_CHARSET sample (first 100 chars): {VIRTUAL_CHARSET[:100]}")
