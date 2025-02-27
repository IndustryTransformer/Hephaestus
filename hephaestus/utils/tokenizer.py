import re
from typing import Dict, List

import pandas as pd


def split_complex_word(word: str) -> List[str]:
    """
    Splits a complex word into its individual parts.

    Args:
        word (str): The complex word to be split.

    Returns:
        list: A list of individual parts of the complex word.

    Example:
        >>> split_complex_word("myComplexWord")
        ['my', 'Complex', 'Word']
    """

    # Step 1: Split by underscore, preserving content within square brackets
    parts = re.split(r"(_|\[.*?\])", word)
    parts = [p for p in parts if p]  # Remove empty strings

    # Step 2: Split camelCase for parts not in square brackets
    def split_camel_case(s: str) -> List[str]:
        """
        Splits a camel case string into a list of words.

        Args:
            s (str): The camel case string to be split.

        Returns:
            list: A list of words obtained from the camel case string.

        Examples:
            >>> split_camel_case("helloWorld")
            ['hello', 'World']
            >>> split_camel_case("thisIsATest")
            ['this', 'Is', 'A', 'Test']
        """

        if s.startswith("[") and s.endswith("]"):
            return [s]
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)

    # Step 3: Apply camelCase splitting to each part and flatten the result
    result = [item for part in parts for item in split_camel_case(part)]

    return result


def convert_object_to_int_tokens(
    df: pd.DataFrame, token_dict: Dict[str, int]
) -> pd.DataFrame:
    """
    Converts object columns to integer tokens using a token dictionary.

    Args:
        df (pandas.DataFrame): The DataFrame containing the object columns to be converted.
        token_dict (dict): A dictionary mapping object values to integer tokens.

    Returns:
        pandas.DataFrame: The DataFrame with object columns converted to integer tokens.
    """

    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(token_dict)
    return df
