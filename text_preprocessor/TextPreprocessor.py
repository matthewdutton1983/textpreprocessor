# Import standard libraries
import os
import re
import inspect
import string
import logging
import csv
from pathlib import Path
from unicodedata import normalize
from typing import Optional, Union, Dict, List, Any

# Import third-party libraries
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
from names_dataset import NameDataset


logging.basicConfig(level=logging.INFO)


class TextPreprocessor:
    def __init__(
        self,
        ignore_spellcheck_word_file_path: Optional[Union[str, Path]] = None,
        custom_sub_csv_file_path: Optional[Union[str, Path]] = None,
        language: str = 'en',
        stemmer: Optional[str] = None,
        lemmatizer: Optional[WordNetLemmatizer] = None,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.pipeline = []

        self._CUSTOM_SUB_CSV_FILE_PATH = custom_sub_csv_file_path or os.path.join(
            os.path.dirname(__file__), './data/custom_substitutions.csv')
        self._IGNORE_SPELLCHECK_WORD_FILE_PATH = ignore_spellcheck_word_file_path or os.path.join(
            os.path.dirname(__file__), './data/ignore_spellcheck_words.txt')

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


    def pipeline_method(func):
        """Decorator to mark methods that can be added to the pipeline"""
        func.is_pipeline_method = True
        return func


    def add_to_pipeline(self, methods):
        """Takes a list of method as an argument and adds it to the pipeline"""
        if not isinstance(methods, list):
            methods = [methods]
        for method in methods:
            if getattr(method.__func__, 'is_pipeline_method', False):
                if method not in self.pipeline:
                    self.pipeline.append(method)
                    self.logger.info(
                        f'{method.__name__} has been successfully added to the pipeline')
                else:
                    self.logger.warning(
                        f'{method.__name__} is already in the pipeline')
            else:
                self.logger.error(
                    f'{method.__name__} is not a pipeline method and cannot be added')


    def remove_from_pipeline(self, methods):
        """Takes a method as an argument and removes it from the pipeline"""
        if not isinstance(methods, list):
            methods = [methods]
        for method in methods:
            if method in self.pipeline:
                self.pipeline.remove(method)
                self.logger.info(
                    f'{method.__name__} has been successfully removed from the pipeline')
            else:
                self.logger.error(f'{method.__name__} is not in the pipeline')


    def view_pipeline(self):
        """Prints the name of each method in the pipeline"""
        if self.pipeline:
            self.logger.info("Current pipeline configuration:")
            for i, method in enumerate(self.pipeline):
                self.logger.info(f'{i+1}: {method.__name__}')
        else:
            self.logger.error('The pipeline is currently empty')


    def execute_pipeline(self, text):
        for method in self.pipeline:
            text = method(text)
        return text


    def clear_pipeline(self):
        if self.pipeline:
            self.pipeline = []
            self.logger.info("All methods have been removed from the pipeline")
        else:
            self.logger.info("The pipeline is already empty")


    ################################## PIPELINE METHODS ##################################

 
    @pipeline_method
    def replace_words(self, input_text: str, replacement_dict: Dict[str, str], 
                      case_sensitive: bool = False) -> str:
        try:
            if case_sensitive:
                regex_pattern = re.compile(r'\b(' + '|'.join(replacement_dict.keys()) + r')\b')
            else:
                regex_pattern = re.compile(r'\b(' + '|'.join(replacement_dict.keys()) + r')\b', re.IGNORECASE)
            return regex_pattern.sub(lambda x: replacement_dict[x.group()], input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def make_lowercase(self, input_text: str) -> str:
        try:
            processed_text = input_text.lower()
            return processed_text
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def make_uppercase(self, input_text: str) -> str:
        try:
            return input_text.upper()
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_numbers(self, input_text: str) -> str:
        try:
            return re.sub('\d+', '', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_list_markers(self, input_text: str) -> str:
        """Remove bullets or numbering in itemized input"""
        try:
            return re.sub('[(\s][0-9a-zA-Z][.)]\s+|[(\s][ivxIVX]+[.)]\s+', ' ', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_urls(self, input_text: str) -> str:
        # TODO: Ability to add custom mask
        try:
            return re.sub('(www|http)\S+', '', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
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
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_duplicate_punctuation(self, input_text: str) -> str:
        try:    
            return re.sub(r'([\!\?\.\,\:\;]){2,}', r'\1', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))
   

    @pipeline_method
    def remove_special_characters(self, input_text: str, remove_unicode: bool = False, 
                                  custom_characters: Optional[List[str]] = None) -> str:
        try:
            if remove_unicode:
                processed_text = re.sub(r'[^\w\s]', '', input_text)
                processed_text = ''.join(char for char in processed_text if ord(char) < 128)
            elif custom_characters is not None:
                for character in custom_characters:
                    processed_text = input_text.replace(character, '')
            else:
                processed_text = re.sub(r'[^\w\s]', '', input_text)
            return processed_text
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))
        
        
    @pipeline_method
    def expand_contractions(self, input_text: str) -> str:
        try:
            return contractions.fix(input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))
    

    @pipeline_method
    def remove_email_addresses(self, input_text: str) -> str:
        # TODO: Ability to add custom mask
        try:
            regex_pattern = '[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}'
            return re.sub(regex_pattern, '', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def check_spelling(self, input_text_or_list: Union[str, List[str]]) -> str:
        # TODO: add acronyms into spell checker to ignore auto correction specified by _IGNORE_SPELLCHECK_WORD_FILE_PATH
        # Make case sensitive
        try:
            spell_checker = SpellChecker(language=self.language, distance=1)
            if self._IGNORE_SPELLCHECK_WORD_FILE_PATH is not None:
                spell_checker.word_frequency.load_text_file(
                    self._IGNORE_SPELLCHECK_WORD_FILE_PATH)

            if input_text_or_list is None or len(input_text_or_list) == 0:
                return ''

            if isinstance(input_text_or_list, str):
                if not input_text_or_list.islower():
                    input_text_or_list = input_text_or_list.lower()
                tokens = word_tokenize(input_text_or_list)
            else:
                tokens = [token.lower() for token in input_text_or_list
                          if token is not None and len(token) > 0]

            misspelled = spell_checker.unknown(tokens)

            for word in misspelled:
                tokens[tokens.index(word)] = spell_checker.correction(word)

            return ' '.join(tokens).strip()
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
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
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def tokenize_words(self, input_text: str) -> List[str]:
        """Converts a text into a list of word tokens"""
        try:
            if input_text is None or len(input_text) == 0:
                return []
            return word_tokenize(input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
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
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))






















# NEED TO REVIEW THE METHODS BELOW
# ADD WARNING WHEN BUILDING PIPELINE THAT IF THE OUTPUT OF A METHOD IS A LIST IT MIGHT SCREW WITH RESULTS




    @pipeline_method
    def remove_phone_numbers(self, input_text: str, use_mask: Optional[bool] = False, custom_mask: Optional[str] = None) -> str:
        try:
            regex_pattern = re.compile(r'(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?')
            if custom_mask:
                use_mask = True
                mask = custom_mask
            else:
                mask = '[PHONE_NUMBER]'
            return regex_pattern.sub(mask if use_mask else '', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def keep_alpha_numeric(self, input_text: str) -> str:
        """Remove any character except alphanumeric characters"""
        try:
            return ' '.join(c for c in input_text if c.isalnum())
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_whitespace(self, input_text: str, remove_duplicate_whitespace: bool = True) -> str:
        """Removes leading, trailing, and (optionally) duplicated whitespace"""
        try:
            if remove_duplicate_whitespace:
                return ''.join(re.split('\s+', input_text.strip(), flags=re.UNICODE))
            return input_text.strip()
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def normalize_unicode(self, input_text: str) -> str:
        """Normalize unicode data to remove umlauts, and accents, etc."""
        try:
            return normalize('NFKD', input_text).encode('ASCII', 'ignore').decode('utf-8')
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def encode_text(self, input_text: str, encoding: str = 'utf-8') -> bytes:
        try:
            if encoding not in ['utf-8', 'ascii']:
                raise ValueError(
                    "Invalid encoding type. Only 'utf-8' and 'ascii' are supported.")
            processed_text = input_text.encode(encoding)
            return processed_text
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_social_security_numbers(self, input_text: str) -> str:
        # TODO: Regex needs to be able to handle spaces
        try:
            regex_pattern = '(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}|(' \
                            '?!219099999|078051120)(?!666|000|9\d{2})\d{3}(?!00)\d{2}(?!0{4})\d{4}'
            return re.sub(regex_pattern, '', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_credit_card_numbers(self, input_text: str) -> str:
        try:
            regex_pattern = '(4[0-9]{12}(?:[0-9]{3})?|(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][' \
                            '0-9]|2720)[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(' \
                            '?:2131|1800|35\d{3})\d{11})'
            return re.sub(regex_pattern, '', input_text)
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def remove_names(self, input_text_or_list: Union[str, List[str]]) -> List[str]:
        try:
            self.logger.info('Loading names dataset ...')
            name_searcher = NameDataset()
            if isinstance(input_text_or_list, str):
                tokens = word_tokenize(input_text_or_list)
                processed_tokens = [token for token in tokens
                                    if (not name_searcher.search_first_name(token)) and
                                    (not name_searcher.search_last_name(token))]
            else:
                processed_tokens = [token for token in input_text_or_list
                                    if (not name_searcher.search_first_name(token)) and
                                    (not name_searcher.search_last_name(token)) and
                                    token is not None and len(token) > 0]
            return processed_tokens
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))
 

    @pipeline_method
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
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
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
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def substitute_token(self, token_list: List[str]) -> List[str]:
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
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    @pipeline_method
    def handle_line_feeds(self, input_text: str, mode: str = 'remove') -> str:
        """Handle line feeds in the input text"""
        try:
            if mode == 'remove':
                processed_text = input_text.replace('\n', '').replace('\r', '')
                return processed_text
            elif mode == 'crlf':
                processed_text = input_text.replace('\n', '\r\n').replace('\r\r\n', '\r\n')
                return processed_text
            elif mode == 'lf':
                processed_text = input_text.replace('\r\n', '\n').replace('\r', '\n')
                return processed_text
            else:
                raise ValueError(
                    f"Invalid mode: '{mode}'. Options are 'remove', 'crlf', and 'lf'.")
        except Exception as e:
            function_name = inspect.currentframe().f_code.co_name
            self.logger.error(f'Error occurred in function {function_name}:', str(e))


    ################################## DEFAULT PIPELINE ##################################


    def load_default_pipeline(self):
        """Adds a set of default methods to the pipeline"""
        if self.pipeline:
            self.clear_pipeline()

        default_methods = [
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


if __name__ == '__main__':
    preprocessor = TextPreprocessor()
    preprocessor.add_to_pipeline([preprocessor.make_lowercase, preprocessor.remove_names])
    preprocessor.view_pipeline()
    text = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
    result = preprocessor.execute_pipeline(text)
    print(result)
