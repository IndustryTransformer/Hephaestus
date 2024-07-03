# %%
import re


def split_complex_word(word):
    # Step 1: Split by asterisk and underscore
    parts = re.split(r"[_]", word)

    # Step 2: Split camelCase
    def split_camel_case(s):
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)

    # Step 3: Apply camelCase splitting to each part and flatten the result
    result = [item for part in parts for item in split_camel_case(part)]

    # Step 4: Convert to lowercase
    result = [item.lower() for item in result]

    return result


# Test the function
test_word = "word_startHere1"
print(split_complex_word(test_word))

# %%
import re


def split_complex_word(word):
    # Step 1: Split by underscore, preserving content within square brackets and asterisks
    parts = re.split(r"(_)", word)
    parts = [p for p in parts if p]  # Remove empty strings

    # Step 2: Split camelCase for parts not in square brackets or containing asterisks
    def split_camel_case(s):
        if s.startswith("[") and s.endswith("]") or "*" in s:
            return [s]
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)

    # Step 3: Apply camelCase splitting to each part and flatten the result
    result = [item for part in parts for item in split_camel_case(part)]

    # Step 4: Convert to lowercase for parts not in square brackets or containing asterisks
    # result = [
    #     item.lower()
    #     if not (item.startswith("[") and item.endswith("]") or "*" in item)
    #     else item
    #     for item in result
    # ]

    return result


def process_word_list(word_list):
    # Flatten the list of complex words
    flattened = [
        word for complex_word in word_list for word in split_complex_word(complex_word)
    ]

    # Use a dictionary to preserve order and remove duplicates
    unique_words = {}
    for word in flattened:
        unique_words.setdefault(word, None)

    # Convert back to list
    return list(unique_words.keys())


# Example usage
complex_words = [
    "word_start[MASK]1",
    "another[Example]2_withMorewords",
    "camel[Case]Word",
]
result = process_word_list(complex_words)
print(result)

# %%
import re


def split_complex_word(word):
    # Step 1: Split by underscore, preserving content within square brackets
    parts = re.split(r"(_|\[.*?\])", word)
    parts = [p for p in parts if p]  # Remove empty strings

    # Step 2: Split camelCase for parts not in square brackets
    def split_camel_case(s):
        if s.startswith("[") and s.endswith("]"):
            return [s]
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)

    # Step 3: Apply camelCase splitting to each part and flatten the result
    result = [item for part in parts for item in split_camel_case(part)]

    return result


def process_word_list(word_list):
    # Flatten the list of complex words
    flattened = [
        word for complex_word in word_list for word in split_complex_word(complex_word)
    ]

    # Use a dictionary to preserve order and remove duplicates
    unique_words = {}
    for word in flattened:
        unique_words.setdefault(word, None)

    # Convert back to list
    return list(unique_words.keys())


# Example usage
complex_words = [
    "word_start[MASK]1",
    "another[Example]2_withMorewords",
    "camel[Case]Word",
]
result = process_word_list(complex_words)
print(result)

# %%
