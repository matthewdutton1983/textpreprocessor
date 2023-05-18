# Import standard libraries
import os
import re
import inspect
import string
import logging
import csv
from pathlib import Path
from unicodedata import normalize
from typing import Optional, Callable, Union, Dict, List, Any

# Import third-party libraries
import contractions
import nltk
import spacy
from abbreviations import schwartz_hearst
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker

# Import project code
from pipeline import Pipeline


logging.basicConfig(level=logging.INFO)


class PreProcessor:
    def __init__(
        self,
        custom_sub_csv_file_path: Optional[Union[str, Path]] = None,
        language: str = 'en',
        stemmer: Optional[str] = None,
        lemmatizer: Optional[WordNetLemmatizer] = None,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.pipeline = None

        self._CUSTOM_SUB_CSV_FILE_PATH = custom_sub_csv_file_path or os.path.join(
            os.path.dirname(__file__), './data/custom_substitutions.csv')

        self.default_lemmatizer = lemmatizer
        self.default_stemmer = None
        if stemmer is not None:
            if stemmer.lower() == 'porter':
                self.default_stemmer = PorterStemmer()
            elif stemmer.lower() == 'snowball':
                self.default_stemmer = SnowballStemmer(language)
            elif stemmer.lower() == 'lancaster':
                self.default_stemmer = LancasterStemmer()
            else:
                raise ValueError(
                    f"Unsupported stemmer '{stemmer}'. Supported stemmers are: 'porter', 'snowball', 'lancaster'")

        supported_languages = ['en', 'es', 'fr', 'pt', 'de', 'ru', 'ar']
        if language not in supported_languages:
            raise ValueError(
                f"Unsupported language '{language}'. Supported language are: {supported_languages}")
        self.language = language

        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)

    def _log_error(self, e: Exception) -> None:
        function_name = inspect.currentframe().f_back.f_code.co_name
        self.logger.error(
            f'Error occurred in function {function_name}: {str(e)}')

    def _pipeline_method(func):
        """Decorator to mark methods that can be added to the pipeline"""
        func.is_pipeline_method = True
        return func
    
    def create_pipeline(self):
        """Create a new Pipeline object"""
        if self.pipeline is not None:
            raise ValueError("A pipeline already exists.")
        self.pipeline = Pipeline()
        return self.pipeline
    
    def delete_pipeline(self):
        """Delete the existing Pipeline object"""
        if self.pipeline is None:
            raise ValueError("No pipeline exists.")
        self.pipeline = None
        
    def add_to_pipeline(self, methods: Union[List[Callable], Callable]) -> None:
        """Takes a list of method as an argument and adds it to the pipeline"""
        if self.pipeline is None:
            raise ValueError("No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.add_methods(methods)

    def remove_from_pipeline(self, methods: Union[List[Callable], Callable]) -> None:
        """Takes a method as an argument and removes it from the pipeline"""
        if self.pipeline is None:
            raise ValueError("No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.remove_methods(methods) 

    def view_pipeline(self) -> None:
        """Prints the name of each method in the pipeline"""
        if self.pipeline is None:
            raise ValueError("No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.view_pipeline()

    def execute_pipeline(self, input_text: str, errors: str = 'continue') -> str:
        if self.pipeline is None:
            raise ValueError("No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.execute_pipeline(input_text, errors)

    def empty_pipeline(self) -> None:
        if self.pipeline is None:
            raise ValueError("No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.clear_pipeline()

    ################################## PIPELINE METHODS ##################################

    @_pipeline_method
    def replace_words(self, input_text: str, replacement_dict: Dict[str, str],
                      case_sensitive: bool = False) -> str:
        try:
            if case_sensitive:
                regex_pattern = re.compile(
                    r'\b(' + '|'.join(replacement_dict.keys()) + r')\b')
            else:
                regex_pattern = re.compile(
                    r'\b(' + '|'.join(replacement_dict.keys()) + r')\b', re.IGNORECASE)
            return regex_pattern.sub(lambda x: replacement_dict[x.group()], input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def make_lowercase(self, input_text: str) -> str:
        try:
            processed_text = input_text.lower()
            return processed_text
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def make_uppercase(self, input_text: str) -> str:
        try:
            return input_text.upper()
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_numbers(self, input_text: str) -> str:
        try:
            return re.sub('\d+', '', input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_list_markers(self, input_text: str) -> str:
        """Remove bullets or numbering in itemized input"""
        try:
            return re.sub('[(\s][0-9a-zA-Z][.)]\s+|[(\s][ivxIVX]+[.)]\s+', ' ', input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_urls(self, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
        try:
            regex_pattern = re.compile(r'(www|http)\S+')
            if custom_mask:
                use_mask = True
                mask = custom_mask
            else:
                mask = '<URL>'
            return regex_pattern.sub(mask if use_mask else '', input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_punctuation(self, input_text: str, punctuations: Optional[str] = None) -> str:
        """
        Removes all punctuations from a string as defined by string.punctuation or a custom list.
        For reference, Python's string.punctuation is equivalent to '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~'
        """
        try:
            if punctuations is None:
                punctuations = string.punctuation
            processed_text = input_text.translate(
                str.maketrans('', '', punctuations))
            return processed_text
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_duplicate_punctuation(self, input_text: str) -> str:
        try:
            return re.sub(r'([\!\?\.\,\:\;]){2,}', r'\1', input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_special_characters(self, input_text: str, remove_unicode: bool = False,
                                  custom_characters: Optional[List[str]] = None) -> str:
        try:
            if remove_unicode:
                processed_text = re.sub(r'[^\w\s]', '', input_text)
                processed_text = ''.join(
                    char for char in processed_text if ord(char) < 128)
            elif custom_characters is not None:
                for character in custom_characters:
                    processed_text = input_text.replace(character, '')
            else:
                processed_text = re.sub(r'[^\w\s]', '', input_text)
            return processed_text
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def expand_contractions(self, input_text: str) -> str:
        try:
            return contractions.fix(input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_email_addresses(self, input_text: str, use_mask: Optional[bool] = True,
                               custom_mask: Optional[str] = None) -> str:
        try:
            regex_pattern = re.compile(
                r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}')
            if custom_mask:
                use_mask = True
                mask = custom_mask
            else:
                mask = '<EMAIL_ADDRESS>'
            return regex_pattern.sub(mask if use_mask else '', input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def check_spelling(self, input_text_or_list: Union[str, List[str]], case_sensitive: bool = True) -> str:
        # How does the user pass in a language?
        # It is making text lower case
        try:
            spell_checker = SpellChecker(language=self.language, distance=1, case_sensitive=case_sensitive)
            
            if input_text_or_list is None or len(input_text_or_list) == 0:
                return ''

            if isinstance(input_text_or_list, str):
                tokens = word_tokenize(input_text_or_list)
            else:
                tokens = [token for token in input_text_or_list 
                          if token is not None and len(token) > 0]

            corrected_tokens = [spell_checker.correction(word) for word in tokens]

            return ' '.join(corrected_tokens).strip()
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_stopwords(self, input_text_or_list: Union[str, List[str]], stop_words: Optional[set] = None) -> List[str]:
        try:
            if stop_words is None:
                stop_words = set(stopwords.words('english'))
            if isinstance(stop_words, list):
                stop_words = set(stop_words)
            if isinstance(input_text_or_list, str):
                tokens = word_tokenize(input_text_or_list)
                processed_tokens = [
                    token for token in tokens if token not in stop_words]
            else:
                processed_tokens = [token for token in input_text_or_list
                                    if (token not in stop_words and token is not None and len(token) > 0)]
            return processed_tokens
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def tokenize_words(self, input_text: str) -> List[str]:
        """Converts a text into a list of word tokens"""
        try:
            if input_text is None or len(input_text) == 0:
                return []
            return word_tokenize(input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def tokenize_sentences(self, input_text: str, tokenizer: Optional[Any] = None) -> List[str]:
        """Converts a text into a list of sentence tokens"""
        try:
            if tokenizer is None:
                tokenizer = PunktSentenceTokenizer()
            if input_text is None or len(input_text) == 0:
                return []
            sentences = tokenizer.tokenize(input_text)
            return sentences
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def normalize_unicode(self, input_text: str) -> str:
        """Normalize unicode data to remove umlauts, and accents, etc."""
        try:
            return normalize('NFKD', input_text).encode('ASCII', 'ignore').decode('utf-8')
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_names(self, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
        try:
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
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_whitespace(self, input_text: str, mode: str = 'strip', keep_duplicates: bool = False) -> str:
        """Remove leading and trailing spaces by default, as well as duplicate whitespace."""
        try:
            if mode == 'leading':
                processed_text = re.sub(
                    r'^\s+', '', input_text, flags=re.UNICODE)
            elif mode == 'trailing':
                processed_text = re.sub(
                    r'\s+$', '', input_text, flags=re.UNICODE)
            elif mode == 'all':
                processed_text = re.sub(
                    r'\s+', '', input_text, flags=re.UNICODE)
            elif mode == 'strip':
                processed_text = re.sub(
                    r'^\s+|\s+$', '', input_text, flags=re.UNICODE)
            else:
                raise ValueError(
                    f"Invalid mode: '{mode}'. Options are 'strip', 'all', 'leading' and 'trailing'.")

            if not keep_duplicates:
                processed_text = ' '.join(
                    re.split('\s+', processed_text, flags=re.UNICODE))

            return processed_text
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_phone_numbers(self, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
        try:
            regex_pattern = re.compile(
                r'(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?(?!\d)')
            mask = custom_mask if custom_mask else '<PHONE_NUMBER>'

            if use_mask:
                return regex_pattern.sub(lambda match: ' ' + mask if match.group().startswith(' ') else mask, input_text)
            else:
                return regex_pattern.sub('', input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def encode_text(self, input_text: str, encoding: str = 'utf-8', errors: str = 'strict') -> bytes:
        try:
            if encoding not in ['utf-8', 'ascii']:
                raise ValueError(
                    "Invalid encoding type. Only 'utf-8' and 'ascii' are supported.")
            if errors not in ['strict', 'ignore', 'replace']:
                raise ValueError(
                    "Invalid error handling strategy. Only 'strict', 'ignore', and 'replace' are supported.")
            processed_text = input_text.encode(encoding, errors=errors)
            return processed_text
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_social_security_numbers(self, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
        """Regex pattern follows the rules for a valid SSN"""
        try:
            regex_pattern = '(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}|(' \
                            '?!219099999|078051120)(?!666|000|9\d{2})\d{3}(?!00)\d{2}(?!0{4})\d{4}'
            mask = custom_mask or '<SOCIAL_SECURITY_NUMBER>'
            return re.sub(regex_pattern, mask if use_mask else '', input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def remove_credit_card_numbers(self, input_text: str, use_mask: Optional[bool] = True, custom_mask: Optional[str] = None) -> str:
        try:
            regex_pattern = '(4[0-9]{12}(?:[0-9]{3})?|(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][' \
                            '0-9]|2720)[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(' \
                            '?:2131|1800|35\d{3})\d{11})'
            mask = '<CREDIT_CARD_NUMBER>' if use_mask else ''
            if custom_mask is not None:
                mask = custom_mask
            return re.sub(regex_pattern, mask, input_text)
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def stem_words(self, input_text_or_list: Union[str, List[str]]) -> List[str]:
        """Stem each token in a text"""
        stemmer = self.default_stemmer if self.default_stemmer is not None else PorterStemmer()
        try:
            if isinstance(input_text_or_list, str):
                tokens = word_tokenize(input_text_or_list)
                processed_tokens = [stemmer.stem(token) for token in tokens]
            else:
                processed_tokens = [stemmer.stem(token) for token in input_text_or_list
                                    if token is not None and len(token) > 0]
            return processed_tokens
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def lemmatize_words(self, input_text_or_list: Union[str, List[str]]) -> List[str]:
        """Lemmatize each token in a text by finding its base form"""
        lemmatizer = self.default_lemmatizer if self.default_lemmatizer is not None else WordNetLemmatizer()
        try:
            if isinstance(input_text_or_list, str):
                tokens = word_tokenize(input_text_or_list)
                processed_tokens = [lemmatizer.lemmatize(
                    token) for token in tokens]
            else:
                processed_tokens = [lemmatizer.lemmatize(token) for token in input_text_or_list
                                    if token is not None and len(token) > 0]
            return processed_tokens
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def substitute_tokens(self, token_list: List[str]) -> List[str]:
        """Substitute each token by another token, e.g. vs -> versus"""
        try:
            with open(self._CUSTOM_SUB_CSV_FILE_PATH, 'r') as f:
                csv_file = csv.reader(f)
                self.sub_dict = dict(csv_file)
            if token_list is None or len(token_list) == 0:
                return []
            processed_tokens = list()
            for token in token_list:
                if token in self.sub_dict:
                    processed_tokens.append(self.sub_dict[token])
                else:
                    processed_tokens.append(token)
            return processed_tokens
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def handle_line_feeds(self, input_text: str, mode: str = 'remove') -> str:
        """Handle line feeds in the input text"""
        try:
            if mode == 'remove':
                processed_text = input_text.replace('\n', '').replace('\r', '')
                return processed_text
            elif mode == 'crlf':
                processed_text = input_text.replace(
                    '\n', '\r\n').replace('\r\r\n', '\r\n')
                return processed_text
            elif mode == 'lf':
                processed_text = input_text.replace(
                    '\r\n', '\n').replace('\r', '\n')
                return processed_text
            else:
                raise ValueError(
                    f"Invalid mode: '{mode}'. Options are 'remove', 'crlf', and 'lf'.")
        except Exception as e:
            self._log_error(e)

    @_pipeline_method
    def find_abbreviations(self, input_text: str) -> List[str]:
        try:
            abbrevs = schwartz_hearst.extract_abbreviation_definition_pairs(input_text)
            return abbrevs
        except Exception as e:
            self.log.error(e)

    ################################## DEFAULT PIPELINE ##################################

    def load_default_pipeline(self) -> None:
        """Adds a set of default methods to the pipeline"""
        if self.pipeline:
            self.clear_pipeline()

        default_methods: List[Callable] = [
            self.tokenize_sentences,
            self.make_lowercase,
            self.remove_stopwords,
            self.lemmatize_words,
            self.check_spelling,
            self.remove_punctuation,
            self.remove_numbers,
        ]

        self.add_to_pipeline(default_methods)
        self.logger.info("Default pipeline loaded.")

    def execute_default_pipeline(self, input_text: str) -> str:
        """Set up and run the default pipeline on the given text"""
        if self.pipeline:
            self.clear_pipeline()
        self.load_default_pipeline()
        return self.execute_pipeline(input_text)
    
