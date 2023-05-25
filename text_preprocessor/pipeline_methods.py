# Import standard libraries
import re
import string
from functools import wraps
from unicodedata import normalize
from typing import Optional, Dict, List

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
def check_spelling(cls, input_text: str, case_sensitive: bool = True, language: str = 'en') -> str:
    """
    This method corrects the spelling of words in a given text using SpellChecker.

    Parameters:
    - input_text (str): The input text to spell check.
    - case_sensitive (bool): A flag indicating whether the spell checker should be case sensitive. Defaults to True.
    - language (str): The language code for the spell checker. Defaults to 'en' (English).

    Returns:
    str: The processed text with corrected spelling.

    Raises:
    ValueError: If an unsupported language is specified.
    """
    supported_languages = ['en', 'es', 'fr', 'pt', 'de', 'ru', 'ar']
    
    if language not in supported_languages:
        raise ValueError(f"Unsupported language '{language}'. Supported languages are: {supported_languages}")
    
    spell_checker = SpellChecker(language=language, case_sensitive=case_sensitive, distance=1)
   
    if not case_sensitive:
        input_text = input_text.lower()

    tokens = word_tokenize(input_text)
    corrected_tokens = []

    for word in tokens:
        if word not in spell_checker:
            correction = spell_checker.correction(word)
            corrected_tokens.append(correction)
        else:
            corrected_tokens.append(word)
    return ' '.join(corrected_tokens).strip()

# @pipeline_method
# def check_spelling(self, input_text_or_list: Union[str, List[str]], case_sensitive: bool = True) -> str:
#     try:
#         spell_checker = SpellChecker(language=self.language, distance=1)
#         if self._IGNORE_SPELLCHECK_WORD_FILE_PATH is not None:
#             spell_checker.word_frequency.load_text_file(
#                 self._IGNORE_SPELLCHECK_WORD_FILE_PATH)

#         if input_text_or_list is None or len(input_text_or_list) == 0:
#             return ''

#         if isinstance(input_text_or_list, str):
#             if not case_sensitive:
#                 input_text_or_list = input_text_or_list.lower()
#             tokens = word_tokenize(input_text_or_list)
#         else:
#             if not case_sensitive:
#                 tokens = [token.lower() for token in input_text_or_list
#                             if token is not None and len(token) > 0]
#             else:
#                 tokens = [token for token in input_text_or_list
#                             if token is not None and len(token) > 0]

#         misspelled = spell_checker.unknown(tokens)

#         for word in misspelled:
#             tokens[tokens.index(word)] = spell_checker.correction(word)

#         return ' '.join(tokens).strip()
#     except Exception as e:
#         self.log_error(e)

@classmethod
@pipeline_method
def encode_text(cls, input_text: str, encoding: str = 'utf-8', errors: str = 'strict') -> str:
    """
    This method encodes given text using a specified encoding.

    Parameters:
    - input_text (str): The input text to encode.
    - encoding (str): The encoding type to use. Defaults to 'utf-8'.
    - errors (str): The error handling strategy to use. Defaults to 'strict'.

    Returns:
    str: The encoded text.

    Raises:
    ValueError: If an unsupported encoding type or error handling strategy is specified.
    """
    if encoding not in ['utf-8', 'ascii']:
        raise ValueError("Invalid encoding type. Only 'utf-8' and 'ascii' are supported.")
    
    if errors not in ['strict', 'ignore', 'replace']:
        raise ValueError("Invalid error handling strategy. Only 'strict', 'ignore', and 'replace' are supported.")

    return input_text.encode(encoding, errors=errors)

@classmethod
@pipeline_method
def expand_contractions(cls, input_text: str) -> str:
    """
    This method expands contractions in given text using the contractions package.

    Parameters:
    - input_text (str): The input text to expand contractions.

    Returns:
    str: The text with expanded contractions.
    """
    return contractions.fix(input_text)

@classmethod
@pipeline_method
def find_abbreviations(cls, input_text: str) -> Dict[str, str]:
    """
    This method identifies abbreviations in given text and returns dictionaries of abbreviations and their definitions.

    Parameters:
    - input_text (str): The input text to identify abbreviations.

    Returns:
    Dict[str, str]: The dictionary containing abbreviations and their definitions.
    """
    abbrevs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=input_text)
    return {abbreviation: definition for abbreviation, definition in abbrevs}

@classmethod
@pipeline_method
def handle_line_feeds(cls, input_text: str, mode: str = 'remove') -> str:
    """
    This method handles line feeds in given text and supports several modes for handling them.

    Parameters:
    - input_text (str): The input text to handle line feeds.
    - mode (str): The mode for handling line feeds. Options are 'remove', 'crlf', and 'lf'. Default is 'remove'.

    Returns:
    str: The processed text with line feeds handled according to the specified mode.
    """
    if mode == 'remove':
        return input_text.replace('\n', '').replace('\r', '')
    elif mode == 'crlf':
        return input_text.replace('\n', '\r\n').replace('\r\r\n', '\r\n')
    elif mode == 'lf':
        return input_text.replace('\r\n', '\n').replace('\r', '\n')
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Options are 'remove', 'crlf', and 'lf'.")

@classmethod
@pipeline_method
def lemmatize_words(cls, input_text: str) -> List[str]:
    """
    This method lemmatizes words in given text, reducing them to their base or dictionary form.

    Parameters:
    - input_text (str): The input text to lemmatize.

    Returns:
    List[str]: The lemmatized words as a list corresponding to the input text.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(input_text)
    return [lemmatizer.lemmatize(token) for token in tokens]


@classmethod
@pipeline_method
def make_lowercase(cls, input_text: str) -> str:
    """
    This method converts given text to lower case.

    Parameters:
    - input_text (str): The input text to convert to lower case.

    Returns:
    str: The text converted to lower case.
    """
    return input_text.lower()

@classmethod
@pipeline_method
def make_uppercase(cls, input_text: str) -> str:
    """
    This method converts given text to upper case.

    Parameters:
    - input_text: str: The input text to convert to upper case.

    Returns:
    str: The text converted to upper case.
    """
    return input_text.upper()


@classmethod
@pipeline_method
def normalize_unicode(cls, input_text: str) -> str:
    """
    This method normalizes unicode characters in given text to remove umlauts, accents, etc.

    Parameters:
    - input_text (str): The input text to normalize unicode.

    Returns:
    str: The text with normalized unicode characters.
    """
    return normalize('NFKD', input_text).encode('ASCII', 'ignore').decode('utf-8')

@classmethod
@pipeline_method
def numbers_to_words(cls, input_text: str) -> str:
    """
    This method converts numbers in the text to their corresponding words.

    Parameters:
    input_text (str): The input text.

    Returns:
    str: The text with numbers converted to words.
    """
    def replace_with_words(match):
        number = match.group(0)
        return num2words(number)
    return re.sub(r'\b\d+\b', replace_with_words, input_text)

@classmethod
@pipeline_method
def replace_words(cls, input_text: str, replacement_dict: Dict[str, str], case_sensitive: bool = False) -> str:
    """
    This method replaces specified words in given text according to a replacement dictionary.

    Parameters:
    - input_text (str): The input text to replace words in.
    - replacement_dict (Dict[str, str]): The dictionary mapping words to their replacements.
    - case_sensitive (bool): Flag indicating whether the replacement should be case-sensitive. Default is False.

    Returns:
    str: The text with specified words replaced according to the replacement dictionary.
    """
    if case_sensitive:
        regex_pattern = re.compile(r'\b(' + '|'.join(replacement_dict.keys()) + r')\b')
    else:
        regex_pattern = re.compile(r'\b(' + '|'.join(replacement_dict.keys()) + r')\b', re.IGNORECASE)
    return regex_pattern.sub(lambda x: replacement_dict[x.group()], input_text)

@classmethod
@pipeline_method
def remove_credit_card_numbers(cls, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
    """
    This method removes credit card numbers from given text.

    Parameters:
    - input_text (str): The input text to remove credit card numbers from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed credit card numbers with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    str: The text with credit card numbers removed.
    """
    regex_pattern = '(4[0-9]{12}(?:[0-9]{3})?|(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][' \
                    '0-9]|2720)[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(' \
                    '?:2131|1800|35\d{3})\d{11})'
    mask = '<CREDIT_CARD_NUMBER>' if use_mask else ''
    if custom_mask is not None:
        mask = custom_mask
    return re.sub(regex_pattern, mask, input_text)

@classmethod
@pipeline_method
def remove_duplicate_punctuation(cls, input_text: str) -> str:
    """
    This method removes duplicate punctuation from given text.

    Parameters:
    - input_text (str): The input text to remove duplicate punctuation from.

    Returns:
    str: The text with duplicate punctuation removed.
    """
    return re.sub(r'([\!\?\.\,\:\;]){2,}', r'\1', input_text)

@classmethod
@pipeline_method
def remove_email_addresses(cls, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
    """
    This method removes email addresses from given text.

    Parameters:
    - input_text (str): The input text to remove email addresses from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed email addresses with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    str: The text with email addresses removed.
    """
    regex_pattern = re.compile(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}')
    if custom_mask:
        use_mask = True
        mask = custom_mask
    else:
        mask = '<EMAIL_ADDRESS>'
    return regex_pattern.sub(mask if use_mask else '', input_text)

@classmethod
@pipeline_method
def remove_list_markers(cls, input_text: str) -> str:
    """
    This method removes list markers (numbering and bullets) from given text.

    Parameters:
    - input_text (str): The input text to remove list markers from.

    Returns:
    str: The text with list markers removed.
    """
    return re.sub(r'(^|\s)[0-9a-zA-Z][.)]\s+|(^|\s)[ivxIVX]+[.)]\s+', ' ', input_text)

@classmethod
@pipeline_method
def remove_names(cls, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
    """
    This method removes names identified as 'PERSON' entities using spaCy's named entity recognition.

    Parameters:
    - input_text (str): The input text to remove names from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed names with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    str: The text with names removed.
    """
    if custom_mask:
        use_mask = True
        mask = custom_mask
    else:
        mask = '<NAME>'

    nlp = spacy.load('en_core_web_sm')
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

    return ''.join([(t if t in string.punctuation else ' ' + t) for t in tokens]).strip()

@classmethod
@pipeline_method
def remove_numbers(cls, input_text: str) -> str:
    """
    This method removes all numbers from given text.

    Parameters:
    - input_text (str): The input text to remove numbers from.

    Returns:
    str: The text with numbers removed.
    """
    return re.sub('\d+', '', input_text)

@classmethod
@pipeline_method
def remove_phone_numbers(cls, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
    """
    This method removes phone numbers from given text.

    Parameters:
    - input_text (str): The input text to remove phone numbers from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed phone numbers with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    str: The text with phone numbers removed.
    """
    regex_pattern = re.compile(r'(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?(?!\d)')
    mask = custom_mask if custom_mask else '<PHONE_NUMBER>'

    if use_mask:
        return regex_pattern.sub(lambda match: ' ' + mask if match.group().startswith(' ') else mask, input_text)
    else:
        return regex_pattern.sub('', input_text)

@classmethod
@pipeline_method
def remove_punctuation(cls, input_text: str, punctuations: Optional[str] = None) -> str:
    """
    This method removes all punctuations from given text.

    Parameters:
    - input_text (str): The input text to remove punctuation from.
    - punctuations (Optional[str]): The specific punctuation characters to remove. If None, removes all punctuations defined by string.punctuation. Default is None.

    Returns:
    str: The text with punctuation removed.
    """
    if punctuations is None:
        punctuations = string.punctuation
    return input_text.translate(str.maketrans('', '', punctuations))

@classmethod
@pipeline_method
def remove_social_security_numbers(cls, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
    """
    This method removes social security numbers from given text.

    Parameters:
    - input_text (str): The input text to remove social security numbers from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed social security numbers with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    str: The text with social security numbers removed.
    """
    regex_pattern = '(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}|(' \
                    '?!219099999|078051120)(?!666|000|9\d{2})\d{3}(?!00)\d{2}(?!0{4})\d{4}'
    mask = custom_mask or '<SOCIAL_SECURITY_NUMBER>'
    return re.sub(regex_pattern, mask if use_mask else '', input_text)

@classmethod
@pipeline_method
def remove_special_characters(cls, input_text: str, remove_unicode: bool = False, custom_characters: Optional[List[str]] = None) -> str:
    """
    This method removes special characters from given text.

    Parameters:
    - input_text (str): The input text to remove special characters from.
    - remove_unicode (bool): Flag indicating whether to remove Unicode characters as well. Default is False.
    - custom_characters (Optional[List[str]]): A list of custom characters to remove. If provided, only these characters will be removed. Default is None.

    Returns:
    str: The text with special characters removed.
    """
    if remove_unicode:
        processed_text = re.sub(r'[^\w\s]', '', input_text)
        processed_text = ''.join(char for char in processed_text if ord(char) < 128)
    elif custom_characters is not None:
        for character in custom_characters:
            processed_text = input_text.replace(character, '')
    else:
        processed_text = re.sub(r'[^\w\s]', '', input_text)
    return processed_text

@classmethod
@pipeline_method
def remove_stopwords(cls, input_text: str, stop_words: Optional[set] = None) -> str:
    """
    This method removes stopwords from given text.

    Parameters:
    - input_text (str): The input text to remove stopwords from.
    - stop_words (Optional[set]): A set of stopwords to remove. If None, uses the default set of English stopwords from NLTK. Default is None.

    Returns:
    str: The list of tokens with stopwords removed.
    """
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    if isinstance(stop_words, list):
        stop_words = set(stop_words)

    tokens = word_tokenize(input_text)
    processed_tokens = [token for token in tokens if token not in stop_words]

    return processed_tokens


@classmethod
@pipeline_method
def remove_urls(cls, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
    """
    This method removes URLs from given text.

    Parameters:
    - input_text (str): The input text to remove URLs from.
    - use_mask (Optional[bool]): Flag indicating whether to replace the removed URLs with a mask. Default is True.
    - custom_mask (Optional[str]): Custom mask to use if `use_mask` is True. Default is None.

    Returns:
    str: The text with URLs removed.
    """
    regex_pattern = re.compile(r'(www|http)\S+')
    
    if custom_mask:
        use_mask = True
        mask = custom_mask
    else:
        mask = '<URL>'

    return regex_pattern.sub(mask if use_mask else '', input_text)

@classmethod
@pipeline_method
def remove_whitespace(cls, input_text: str, mode: str = 'strip', keep_duplicates: bool = False) -> str:
    """
    This method removes whitespace from given text according to specified mode.

    Parameters:
    - input_text (str): The input text to remove whitespace from.
    - mode (str): The mode for removing whitespace. Options are 'strip', 'all', 'leading', and 'trailing'. Default is 'strip'.
    - keep_duplicates (bool): Flag indicating whether to keep duplicate whitespace. Default is False.

    Returns:
    str: The text with whitespace removed.
    """
    if mode == 'leading':
        processed_texts = [re.sub(r'^\s+', '', text, flags=re.UNICODE) for text in input_text]
    elif mode == 'trailing':
        processed_texts = [re.sub(r'\s+$', '', text, flags=re.UNICODE) for text in input_text]
    elif mode == 'all':
        processed_texts = [re.sub(r'\s+', '', text, flags=re.UNICODE) for text in input_text]
    elif mode == 'strip':
        processed_texts = [re.sub(r'^\s+|\s+$', '', text, flags=re.UNICODE) for text in input_text]
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Options are 'strip', 'all', 'leading', and 'trailing'.")

    if not keep_duplicates:
        processed_texts = [' '.join(re.split('\s+', text, flags=re.UNICODE)) for text in processed_texts]

    return processed_texts


@classmethod
@pipeline_method
def stem_words(cls, input_text: str, stemmer: Optional[str] = None) -> List[str]:
    """
    This method stems words in given text, reducing them to their word stem.

    Parameters:
    - input_text (str): The input text to stem words.
    - stemmer (Optional[str]): The stemmer algorithm to use. Options are 'snowball', 'porter', and 'lancaster'. Default is 'porter'.

    Returns:
    List[str]: The list of stemmed tokens.
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
        raise ValueError(f"Unsupported stemmer '{stemmer}'. Supported stemmers are: {', '.join(supported_stemmers.keys())}")

    tokens = word_tokenize(input_text)
    return [supported_stemmers[stemmer].stem(token) for token in tokens]

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
    return word_tokenize(input_text)
