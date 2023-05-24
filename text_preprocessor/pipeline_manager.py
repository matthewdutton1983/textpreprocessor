# Import standard libraries
import inspect
import logging
from typing import Callable, Union, List

# Import third-party libraries
import nltk

# Import project code
from text_preprocessor.pipeline import Pipeline
from text_preprocessor import pipeline_methods


logging.basicConfig(level=logging.INFO)


class PreProcessor():
    # Attach methods
    check_spelling = pipeline_methods.check_spelling
    encode_text = pipeline_methods.encode_text
    expand_contractions = pipeline_methods.expand_contractions
    find_abbreviations = pipeline_methods.find_abbreviations
    handle_line_feeds = pipeline_methods.handle_line_feeds
    lemmatize_words = pipeline_methods.lemmatize_words
    make_lowercase = pipeline_methods.make_lowercase
    make_uppercase = pipeline_methods.make_uppercase
    normalize_unicode = pipeline_methods.normalize_unicode
    remove_credit_card_numbers = pipeline_methods.remove_credit_card_numbers
    remove_duplicate_punctuation = pipeline_methods.remove_duplicate_punctuation
    remove_email_addresses = pipeline_methods.remove_email_addresses
    remove_list_markers = pipeline_methods.remove_list_markers
    remove_names = pipeline_methods.remove_names
    remove_numbers = pipeline_methods.remove_numbers
    remove_phone_numbers = pipeline_methods.remove_phone_numbers
    remove_punctuation = pipeline_methods.remove_punctuation
    remove_social_security_numbers = pipeline_methods.remove_social_security_numbers
    remove_special_characters = pipeline_methods.remove_special_characters
    remove_stopwords = pipeline_methods.remove_stopwords
    remove_urls = pipeline_methods.remove_urls  
    remove_whitespace = pipeline_methods.remove_whitespace
    replace_words = pipeline_methods.replace_words
    stem_words = pipeline_methods.stem_words
    tokenize_sentences = pipeline_methods.tokenize_sentences
    tokenize_words = pipeline_methods.tokenize_words

    def __init__(self):
        super().__init__()
        
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.pipeline = None

    def _log_error(self, e: Exception) -> None:
        function_name = inspect.currentframe().f_back.f_code.co_name
        self.logger.error(f'Error occurred in function {function_name}: {str(e)}')

    def create_pipeline(self, load_defaults: bool = False) -> None:
        """Create a new Pipeline object"""
        if self.pipeline is not None:
            raise ValueError("A pipeline already exists.")
        self.pipeline = Pipeline()

        if load_defaults:
            default_methods: List[Callable] = [
                self.tokenize_sentences,
                self.make_lowercase,
                self.remove_punctuation,
                self.remove_stopwords,
                self.lemmatize_words,
                self.handle_line_feeds,
                self.remove_whitespace,
                self.check_spelling,
            ]
            self.pipeline._load_default_pipeline(default_methods)

        return self.pipeline

    def delete_pipeline(self) -> None:
        """Delete the existing Pipeline object"""
        if self.pipeline is None:
            raise ValueError("No pipeline exists.")
        self.pipeline = None

    def add_to_pipeline(self, methods: Union[List[Callable], Callable]) -> None:
        """Takes a list of method as an argument and adds it to the pipeline"""
        if self.pipeline is None:
            raise ValueError(
                "No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.add_methods(methods)

    def remove_from_pipeline(self, methods: Union[List[Callable], Callable]) -> None:
        """Takes a method as an argument and removes it from the pipeline"""
        if self.pipeline is None:
            raise ValueError(
                "No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.remove_methods(methods)

    def empty_pipeline(self) -> None:
        if self.pipeline is None:
            raise ValueError(
                "No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.clear_pipeline()

    def execute_pipeline(self, input_text: str, errors: str = 'continue') -> str:
        if self.pipeline is None:
            raise ValueError(
                "No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        return self.pipeline.execute_pipeline(input_text, errors)
