import random


def replace_characters(text):
    replacements = {
        'a': '@',
        'e': '3',
        'i': '1',
        'o': '0',
        's': '$'
    }

    return ''.join([replacements.get(c, c) for c in text])


def random_insertion(text):
    chars = list(text)
    if len(chars) > 3:
        idx = random.randint(0, len(chars) - 1)
        chars.insert(idx, random.choice(['#', '!', '%']))
    return ''.join(chars)


def add_noise(text):
    text = replace_characters(text)
    text = random_insertion(text)
    return text