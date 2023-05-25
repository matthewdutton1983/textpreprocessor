# Import standard libraries
import re
import string
from functools import wraps
from unicodedata import normalize
from typing import Optional, Union, Dict, List

# Import third-party libraries
import contractions
import spacy
from abbreviations import schwartz_hearst
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from num2words import num2words
from spellchecker import SpellChecker


def pipeline_method(func):
    """Decorator to mark methods that can be added to the pipeline"""
    func.is_pipeline_method = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise Exception(f"{func.__name__}: {str(e)}")

    return wrapper


@classmethod
@pipeline_method
def check_spelling(cls, input_text_or_list: Union[str, List[str]], case_sensitive: bool = True, language: str = 'en') -> Union[str, List[str]]:
    """
    This method corrects the spelling of words in a given text or list of texts using SpellChecker.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to spell check.
    - case_sensitive (bool): A flag indicating whether the spell checker should be case sensitive. Defaults to True.
    - language (str): The language code for the spell checker. Defaults to 'en' (English).

    Returns:
    Union[str, List[str]]: The processed text or list of processed texts with corrected spelling.

    Raises:
    ValueError: If an unsupported language is specified.
    """
    supported_languages = ['en', 'es', 'fr', 'pt', 'de', 'ru', 'ar']
    if language not in supported_languages:
        raise ValueError(
            f"Unsupported language '{language}'. Supported languages are: {supported_languages}")

    spell_checker = SpellChecker(
        language=language, case_sensitive=case_sensitive)

    if isinstance(input_text_or_list, str):
        input_text_or_list = [input_text_or_list]

    processed_texts = []
    for input_text in input_text_or_list:
        tokens = word_tokenize(input_text)
        corrected_tokens = []
        for word in tokens:
            if word not in spell_checker:
                correction = spell_checker.correction(word)
                corrected_tokens.append(correction)
            else:
                corrected_tokens.append(word)
        corrected_text = ' '.join(corrected_tokens).strip()
        processed_texts.append(corrected_text)

    if len(processed_texts) == 1:
        return processed_texts[0]
    else:
        return processed_texts


@classmethod
@pipeline_method
def encode_text(cls, input_text_or_list: Union[str, List[str]],
                encoding: str = 'utf-8', errors: str = 'strict') -> Union[bytes, List[bytes]]:
    """
    This method encodes given text or list of text using a specified encoding.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to encode.
    - encoding (str): The encoding type to use. Defaults to 'utf-8'.
    - errors (str): The error handling strategy to use. Defaults to 'strict'.

    Returns:
    Union[bytes, List[bytes]]: The encoded text or list of encoded texts.

    Raises:
    ValueError: If an unsupported encoding type or error handling strategy is specified.
    """
    if encoding not in ['utf-8', 'ascii']:
        raise ValueError(
            "Invalid encoding type. Only 'utf-8' and 'ascii' are supported.")
    if errors not in ['strict', 'ignore', 'replace']:
        raise ValueError(
            "Invalid error handling strategy. Only 'strict', 'ignore', and 'replace' are supported.")

    if isinstance(input_text_or_list, str):
        input_text_or_list = [input_text_or_list]

    processed_texts = []
    for input_text in input_text_or_list:
        processed_text = input_text.encode(encoding, errors=errors)
        processed_texts.append(processed_text)

    if len(processed_texts) == 1:
        return processed_texts[0]
    else:
        return processed_texts


@classmethod
@pipeline_method
def expand_contractions(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method expands contractions in given text or list of text using the contractions package.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to expand contractions.

    Returns:
    Union[str, List[str]]: The text or list of texts with expanded contractions.
    """
    if isinstance(input_text_or_list, str):
        input_text_or_list = [input_text_or_list]

    processed_texts = []
    for input_text in input_text_or_list:
        processed_text = contractions.fix(input_text)
        processed_texts.append(processed_text)

    if len(processed_texts) == 1:
        return processed_texts[0]
    else:
        return processed_texts


@classmethod
@pipeline_method
def find_abbreviations(cls, input_text_or_list: Union[str, List[str]]) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """
    This method identifies abbreviations in given text or list of text and returns dictionaries of abbreviations and their definitions.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to identify abbreviations.

    Returns:
    Union[Dict[str, str], List[Dict[str, str]]]: The dictionary or list of dictionaries containing abbreviations and their definitions.
    """
    if isinstance(input_text_or_list, str):
        input_text_or_list = [input_text_or_list]

    processed_texts = []
    for input_text in input_text_or_list:
        abbrevs = schwartz_hearst.extract_abbreviation_definition_pairs(
            doc_text=input_text)
        abbreviations_dict = {
            abbreviation: definition for abbreviation, definition in abbrevs}
        processed_texts.append(abbreviations_dict)

    if len(processed_texts) == 1:
        return processed_texts[0]
    else:
        return processed_texts


@classmethod
@pipeline_method
def handle_line_feeds(cls, input_text_or_list: Union[str, List[str]], mode: str = 'remove') -> Union[str, List[str]]:
    """
    This method handles line feeds in given text or list of text and supports several modes for handling them.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to handle line feeds.
    - mode (str): The mode for handling line feeds. Options are 'remove', 'crlf', and 'lf'. Default is 'remove'.

    Returns:
    Union[str, List[str]]: The processed text or list of processed texts with line feeds handled according to the specified mode.
    """
    if isinstance(input_text_or_list, str):
        input_text_or_list = [input_text_or_list]

    processed_texts = []
    for input_text in input_text_or_list:
        if mode == 'remove':
            processed_text = input_text.replace('\n', '').replace('\r', '')
        elif mode == 'crlf':
            processed_text = input_text.replace(
                '\n', '\r\n').replace('\r\r\n', '\r\n')
        elif mode == 'lf':
            processed_text = input_text.replace(
                '\r\n', '\n').replace('\r', '\n')
        else:
            raise ValueError(
                f"Invalid mode: '{mode}'. Options are 'remove', 'crlf', and 'lf'.")
        processed_texts.append(processed_text)

    if len(processed_texts) == 1:
        return processed_texts[0]
    else:
        return processed_texts


@classmethod
@pipeline_method
def lemmatize_words(cls, input_text_or_list: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
    """
    This method lemmatizes words in given text or list of text, reducing them to their base or dictionary form.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to lemmatize.

    Returns:
    Union[List[str], List[List[str]]]: The lemmatized words as a list or list of lists corresponding to the input text or list of texts.
    """
    lemmatizer = WordNetLemmatizer()

    if isinstance(input_text_or_list, str):
        tokens = word_tokenize(input_text_or_list)
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        processed_tokens = []
        for input_text in input_text_or_list:
            tokens = word_tokenize(input_text)
            processed_tokens.append(
                [lemmatizer.lemmatize(token) for token in tokens])

    return processed_tokens


@classmethod
@pipeline_method
def make_lowercase(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method converts given text or list of text to lower case.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to convert to lower case.

    Returns:
    Union[str, List[str]]: The text or list of texts converted to lower case.
    """
    if isinstance(input_text_or_list, str):
        return input_text_or_list.lower()
    else:
        return [text.lower() for text in input_text_or_list]


@classmethod
@pipeline_method
def make_uppercase(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method converts given text or list of text to upper case.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to convert to upper case.

    Returns:
    Union[str, List[str]]: The text or list of texts converted to upper case.
    """
    if isinstance(input_text_or_list, str):
        return input_text_or_list.upper()
    else:
        return [text.upper() for text in input_text_or_list]


@classmethod
@pipeline_method
def normalize_unicode(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method normalizes unicode characters in given text or list of text to remove umlauts, accents, etc.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to normalize unicode.

    Returns:
    Union[str, List[str]]: The text or list of texts with normalized unicode characters.
    """
    if isinstance(input_text_or_list, str):
        return normalize('NFKD', input_text_or_list).encode('ASCII', 'ignore').decode('utf-8')
    else:
        return [normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8') for text in input_text_or_list]


@classmethod
@pipeline_method
def numbers_to_words(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method converts numbers in the text or list of text to their corresponding words.

    Parameters:
    input_text_or_list (str or List[str]): The input text or list of text.

    Returns:
    str or List[str]: The text or list of text with numbers converted to words.
    """
    def replace_with_words(match):
        number = match.group(0)
        return num2words(number)

    if isinstance(input_text_or_list, str):
        return re.sub(r'\b\d+\b', replace_with_words, input_text_or_list)
    else:
        return [re.sub(r'\b\d+\b', replace_with_words, text) for text in input_text_or_list]


@classmethod
@pipeline_method
def replace_words(cls, input_text_or_list: Union[str, List[str]], replacement_dict: Dict[str, str], case_sensitive: bool = False) -> Union[str, List[str]]:
    """
    This method replaces specified words in given text or list of text according to a replacement dictionary.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to replace words in.
    - replacement_dict (Dict[str, str]): The dictionary mapping words to their replacements.
    - case_sensitive (bool): Flag indicating whether the replacement should be case-sensitive. Default is False.

    Returns:
    Union[str, List[str]]: The text or list of texts with specified words replaced according to the replacement dictionary.
    """
    if isinstance(input_text_or_list, str):
        input_text_or_list = [input_text_or_list]

    processed_texts = []
    for input_text in input_text_or_list:
        if case_sensitive:
            regex_pattern = re.compile(
                r'\b(' + '|'.join(replacement_dict.keys()) + r')\b')
        else:
            regex_pattern = re.compile(
                r'\b(' + '|'.join(replacement_dict.keys()) + r')\b', re.IGNORECASE)
        processed_text = regex_pattern.sub(
            lambda x: replacement_dict[x.group()], input_text)
        processed_texts.append(processed_text)

    if len(processed_texts) == 1:
        return processed_texts[0]
    else:
        return processed_texts


@classmethod
@pipeline_method
def remove_credit_card_numbers(cls, input_text_or_list: Union[str, List[str]], use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> Union[str, List[str]]:
    """
    This method removes credit card numbers from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove credit card numbers from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed credit card numbers with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts with credit card numbers removed.
    """
    regex_pattern = '(4[0-9]{12}(?:[0-9]{3})?|(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][' \
                    '0-9]|2720)[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(' \
                    '?:2131|1800|35\d{3})\d{11})'
    mask = '<CREDIT_CARD_NUMBER>' if use_mask else ''
    if custom_mask is not None:
        mask = custom_mask

    if isinstance(input_text_or_list, str):
        return re.sub(regex_pattern, mask, input_text_or_list)
    else:
        return [re.sub(regex_pattern, mask, text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_duplicate_punctuation(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method removes duplicate punctuation from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove duplicate punctuation from.

    Returns:
    Union[str, List[str]]: The text or list of texts with duplicate punctuation removed.
    """
    if isinstance(input_text_or_list, str):
        return re.sub(r'([\!\?\.\,\:\;]){2,}', r'\1', input_text_or_list)
    else:
        return [re.sub(r'([\!\?\.\,\:\;]){2,}', r'\1', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_email_addresses(cls, input_text_or_list: Union[str, List[str]], use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> Union[str, List[str]]:
    """
    This method removes email addresses from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove email addresses from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed email addresses with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts with email addresses removed.
    """
    regex_pattern = re.compile(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}')
    if custom_mask:
        use_mask = True
        mask = custom_mask
    else:
        mask = '<EMAIL_ADDRESS>'

    if isinstance(input_text_or_list, str):
        return regex_pattern.sub(mask if use_mask else '', input_text_or_list)
    else:
        return [regex_pattern.sub(mask if use_mask else '', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_list_markers(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method removes list markers (numbering and bullets) from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove list markers from.

    Returns:
    Union[str, List[str]]: The text or list of texts with list markers removed.
    """
    if isinstance(input_text_or_list, str):
        return re.sub(r'(^|\s)[0-9a-zA-Z][.)]\s+|(^|\s)[ivxIVX]+[.)]\s+', ' ', input_text_or_list)
    else:
        return [re.sub(r'(^|\s)[0-9a-zA-Z][.)]\s+|(^|\s)[ivxIVX]+[.)]\s+', ' ', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_names(cls, input_text_or_list: Union[str, List[str]], use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> Union[str, List[str]]:
    """
    This method removes names identified as 'PERSON' entities using spaCy's named entity recognition.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove names from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed names with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts with names removed.
    """
    if custom_mask:
        use_mask = True
        mask = custom_mask
    else:
        mask = '<NAME>'

    nlp = spacy.load('en_core_web_sm')

    if isinstance(input_text_or_list, str):
        doc = nlp(input_text_or_list)
        tokens = []

        index = 0

        while index < len(doc):
            token = doc[index]
            if token.ent_type_ == 'PERSON':
                if use_mask:
                    tokens.append(mask)
                while index < len(doc) and doc[index].ent_type_ == 'PERSON':
                    index += 1
            else:
                tokens.append(token.text)
                index += 1

        return ''.join([(t if t in string.punctuation else ' ' + t) for t in tokens]).strip()
    else:
        processed_texts = []
        for input_text in input_text_or_list:
            doc = nlp(input_text)
            tokens = []

            index = 0

            while index < len(doc):
                token = doc[index]
                if token.ent_type_ == 'PERSON':
                    if use_mask:
                        tokens.append(mask)
                    while index < len(doc) and doc[index].ent_type_ == 'PERSON':
                        index += 1
                else:
                    tokens.append(token.text)
                    index += 1

            processed_text = ''.join(
                [(t if t in string.punctuation else ' ' + t) for t in tokens]).strip()
            processed_texts.append(processed_text)

        return processed_texts


@classmethod
@pipeline_method
def remove_numbers(cls, input_text_or_list: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    This method removes all numbers from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove numbers from.

    Returns:
    Union[str, List[str]]: The text or list of texts with numbers removed.
    """
    if isinstance(input_text_or_list, str):
        return re.sub('\d+', '', input_text_or_list)
    else:
        return [re.sub('\d+', '', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_phone_numbers(cls, input_text_or_list: Union[str, List[str]], use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> Union[str, List[str]]:
    """
    This method removes phone numbers from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove phone numbers from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed phone numbers with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts with phone numbers removed.
    """
    regex_pattern = re.compile(
        r'(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?(?!\d)')
    mask = custom_mask if custom_mask else '<PHONE_NUMBER>'

    if use_mask:
        if isinstance(input_text_or_list, str):
            return regex_pattern.sub(lambda match: ' ' + mask if match.group().startswith(' ') else mask, input_text_or_list)
        else:
            return [regex_pattern.sub(lambda match: ' ' + mask if match.group().startswith(' ') else mask, text) for text in input_text_or_list]
    else:
        if isinstance(input_text_or_list, str):
            return regex_pattern.sub('', input_text_or_list)
        else:
            return [regex_pattern.sub('', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_punctuation(cls, input_text_or_list: Union[str, List[str]], punctuations: Optional[str] = None) -> Union[str, List[str]]:
    """
    This method removes all punctuations from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove punctuation from.
    - punctuations (Optional[str]): The specific punctuation characters to remove. If None, removes all punctuations defined by string.punctuation. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts with punctuation removed.
    """
    if punctuations is None:
        punctuations = string.punctuation

    if isinstance(input_text_or_list, str):
        return input_text_or_list.translate(str.maketrans('', '', punctuations))
    else:
        return [text.translate(str.maketrans('', '', punctuations)) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_social_security_numbers(cls, input_text_or_list: Union[str, List[str]], use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> Union[str, List[str]]:
    """
    This method removes social security numbers from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove social security numbers from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed social security numbers with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts with social security numbers removed.
    """
    regex_pattern = '(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}|(' \
                    '?!219099999|078051120)(?!666|000|9\d{2})\d{3}(?!00)\d{2}(?!0{4})\d{4}'
    mask = custom_mask or '<SOCIAL_SECURITY_NUMBER>'

    if isinstance(input_text_or_list, str):
        return re.sub(regex_pattern, mask if use_mask else '', input_text_or_list)
    else:
        return [re.sub(regex_pattern, mask if use_mask else '', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_special_characters(cls, input_text_or_list: Union[str, List[str]], remove_unicode: bool = False, custom_characters: Optional[List[str]] = None) -> Union[str, List[str]]:
    """
    This method removes special characters from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove special characters from.
    - remove_unicode (bool): Flag indicating whether to remove Unicode characters as well. Default is False.
    - custom_characters (Optional[List[str]]): A list of custom characters to remove. If provided, only these characters will be removed. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts with special characters removed.
    """
    if remove_unicode:
        if isinstance(input_text_or_list, str):
            processed_text = re.sub(r'[^\w\s]', '', input_text_or_list)
            processed_text = ''.join(
                char for char in processed_text if ord(char) < 128)
            return processed_text
        else:
            processed_texts = []
            for input_text in input_text_or_list:
                processed_text = re.sub(r'[^\w\s]', '', input_text)
                processed_text = ''.join(
                    char for char in processed_text if ord(char) < 128)
                processed_texts.append(processed_text)
            return processed_texts
    elif custom_characters is not None:
        if isinstance(input_text_or_list, str):
            for character in custom_characters:
                input_text_or_list = input_text_or_list.replace(character, '')
            return input_text_or_list
        else:
            processed_texts = []
            for input_text in input_text_or_list:
                for character in custom_characters:
                    input_text = input_text.replace(character, '')
                processed_texts.append(input_text)
            return processed_texts
    else:
        if isinstance(input_text_or_list, str):
            return re.sub(r'[^\w\s]', '', input_text_or_list)
        else:
            return [re.sub(r'[^\w\s]', '', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_stopwords(cls, input_text_or_list: Union[str, List[str]], stop_words: Optional[set] = None) -> Union[List[str], List[List[str]]]:
    """
    This method removes stopwords from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove stopwords from.
    - stop_words (Optional[set]): A set of stopwords to remove. If None, uses the default set of English stopwords from NLTK. Default is None.

    Returns:
    Union[List[str], List[List[str]]]: The list of tokens or list of lists of tokens with stopwords removed.
    """
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    if isinstance(stop_words, list):
        stop_words = set(stop_words)

    if isinstance(input_text_or_list, str):
        tokens = word_tokenize(input_text_or_list)
        processed_tokens = [
            token for token in tokens if token not in stop_words]
        return processed_tokens
    else:
        processed_tokens = []
        for input_text in input_text_or_list:
            tokens = word_tokenize(input_text)
            processed_tokens.append(
                [token for token in tokens if token not in stop_words])
        return processed_tokens


@classmethod
@pipeline_method
def remove_urls(cls, input_text_or_list: Union[str, List[str]], use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> Union[str, List[str]]:
    """
    This method removes URLs from given text or list of text.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove URLs from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed URLs with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    Union[str, List[str]]: The text or list of texts withURLs removed.
    """
    regex_pattern = re.compile(r'(www|http)\S+')
    if custom_mask:
        use_mask = True
        mask = custom_mask
    else:
        mask = '<URL>'

    if isinstance(input_text_or_list, str):
        return regex_pattern.sub(mask if use_mask else '', input_text_or_list)
    else:
        return [regex_pattern.sub(mask if use_mask else '', text) for text in input_text_or_list]


@classmethod
@pipeline_method
def remove_whitespace(cls, input_text_or_list: Union[str, List[str]], mode: str = 'strip', keep_duplicates: bool = False) -> Union[str, List[str]]:
    """
    This method removes whitespace from given text or list of text according to specified mode.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to remove whitespace from.
    - mode (str): The mode for removing whitespace. Options are 'strip', 'all', 'leading', and 'trailing'. Default is 'strip'.
    - keep_duplicates (bool): Flag indicating whether to keep duplicate whitespace. Default is False.

    Returns:
    Union[str, List[str]]: The text or list of texts with whitespace removed.
    """
    if mode == 'leading':
        processed_texts = [re.sub(r'^\s+', '', text, flags=re.UNICODE)
                           for text in input_text_or_list]
    elif mode == 'trailing':
        processed_texts = [re.sub(r'\s+$', '', text, flags=re.UNICODE)
                           for text in input_text_or_list]
    elif mode == 'all':
        processed_texts = [re.sub(r'\s+', '', text, flags=re.UNICODE)
                           for text in input_text_or_list]
    elif mode == 'strip':
        processed_texts = [re.sub(r'^\s+|\s+$', '', text, flags=re.UNICODE)
                           for text in input_text_or_list]
    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Options are 'strip', 'all', 'leading', and 'trailing'.")

    if not keep_duplicates:
        processed_texts = [
            ' '.join(re.split('\s+', text, flags=re.UNICODE)) for text in processed_texts]

    if len(processed_texts) == 1:
        return processed_texts[0]
    else:
        return processed_texts


@classmethod
@pipeline_method
def stem_words(cls, input_text_or_list: Union[str, List[str]], stemmer: Optional[str] = None) -> Union[List[str], List[List[str]]]:
    """
    This method stems words in given text or list of text, reducing them to their word stem.

    Parameters:
    - input_text_or_list (Union[str, List[str]]): The input text or list of texts to stem words.
    - stemmer (Optional[str]): The stemmer algorithm to use. Options are 'snowball', 'porter', and 'lancaster'. Default is 'porter'.

    Returns:
    Union[List[str], List[List[str]]]: The list of stemmed tokens or list of lists of stemmed tokens.
    """
    supported_stemmers = {
        'snowball': SnowballStemmer('english'),
        'porter': PorterStemmer(),
        'lancaster': LancasterStemmer()
    }
    if stemmer is None:
        stemmer = 'porter'
    stemmer = stemmer.lower()

    if stemmer not in supported_stemmers:
        raise ValueError(
            f"Unsupported stemmer '{stemmer}'. Supported stemmers are: {', '.join(supported_stemmers.keys())}")

    if isinstance(input_text_or_list, str):
        tokens = word_tokenize(input_text_or_list)
        processed_tokens = [supported_stemmers[stemmer].stem(
            token) for token in tokens]
        return processed_tokens
    else:
        processed_tokens = []
        for input_text in input_text_or_list:
            tokens = word_tokenize(input_text)
            processed_tokens.append(
                [supported_stemmers[stemmer].stem(token) for token in tokens])
        return processed_tokens


@classmethod
@pipeline_method
def tokenize_sentences(cls, input_text: str) -> List[str]:
    """
    This method tokenizes a given text into sentences.

    Parameters:
    - input_text (str): The input text to tokenize into sentences.

    Returns:
    List[str]: The list of sentences in the input text.
    """
    if input_text is None or len(input_text) == 0:
        return []
    return [str(s) for s in sent_tokenize(input_text)]


@classmethod
@pipeline_method
def tokenize_words(cls, input_text: str) -> List[str]:
    """
    This method tokenizes a given text into words.

    Parameters:
    - input_text (str): The input text to tokenize into words.

    Returns:
    List[str]: The list of words in the input text.
    """
    if input_text is None or len(input_text) == 0:
        return []
    return word_tokenize(input_text)
