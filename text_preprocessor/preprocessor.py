# Import standard libraries
import inspect
import logging
from functools import wraps
from typing import Callable, Optional, Union, List

# Import project code
from text_preprocessor import pipeline_methods

logging.basicConfig(level=logging.INFO)


class PreProcessor():
    def __init__(self, default_pipeline: bool = False):
        """
        Initializes the PreProcessor class.

        Parameters
        ----------
        load_defaults : bool, optional
            If True, a default set of methods is loaded into the pipeline.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.pipeline = []

        if default_pipeline:
            default_methods: List[Callable] = [
                self.tokenize_sentences,
                self.expand_contractions,
                self.make_lowercase,
                self.remove_punctuation,
                self.remove_stopwords,
                self.lemmatize_words,
                self.handle_line_feeds,
                self.remove_whitespace,
                self.check_spelling,
            ]
            self.add_methods(default_methods)

    @property
    def total_methods(self) -> int:
        """Returns the number of methods in the pipeline"""
        return len(self.pipeline)

    @property
    def list_methods(self) -> List[Callable]:
        """Returns a list of methods in the pipeline"""
        return self.pipeline

    @property
    def is_empty(self) -> bool:
        """Checks if the pipeline is empty"""
        return len(self.pipeline) == 0

    @property
    def last_method(self) -> Optional[Callable]:
        """Returns the last method added to the pipeline"""
        if self.pipeline:
            return self.pipeline[-1]
        else:
            return None

    @property
    def first_method(self) -> Optional[Callable]:
        """Returns the first method added to the pipeline"""
        if self.pipeline:
            return self.pipeline[0]
        else:
            return None

    def _log_error(self, e: Exception) -> None:
        """
        Logs the exception error.

        Parameters
        ----------
        e : Exception
            The exception to be logged.
        """
        function_name = inspect.currentframe().f_back.f_code.co_name
        self.logger.error(
            f'Error occurred in function {function_name}: {str(e)}')

    def _is_iterable(self, obj):
        """Check if the object is iterable"""
        try:
            iter(obj)
        except TypeError:
            return False
        return True

    def _prioritize_string_methods(self) -> None:
        """
        Gives priority to string methods in the pipeline.
        """
        str_methods = [
            method for method in self.pipeline if method.__annotations__.get('return') == str
        ]
        other_methods = [
            method for method in self.pipeline if method.__annotations__.get('return') != str
        ]

        self.pipeline = str_methods + other_methods

    def add_methods(self, methods: Union[List[Callable], Callable]) -> None:
        """
        Adds the given methods to the pipeline.

        Parameters
        ----------
        methods : Union[List[Callable], Callable]
            The method(s) to be added to the pipeline.
        """
        if not self._is_iterable(methods):
            methods = [methods]

        for method in methods:
            if getattr(method, "is_pipeline_method", False):
                if method not in self.pipeline:
                    self.pipeline.append(method)
                    self.logger.info(
                        f'{method.__name__} has been added to the pipeline.')
                else:
                    self.logger.warning(
                        f'{method.__name__} is already in the pipeline.')
            else:
                self.logger.warning(
                    f'{method.__name__} is not a valid pipeline method.')

    def remove_methods(self, methods: Union[List[Callable], Callable]) -> None:
        """
        Removes the given methods from the pipeline.

        Parameters
        ----------
        methods : Union[List[Callable], Callable]
            The method(s) to be removed from the pipeline.
        """
        if not self._is_iterable(methods):
            methods = [methods]

        for method in methods:
            if method in self.pipeline:
                self.pipeline.remove(method)
                self.logger.info(
                    f'{method.__name__} has been removed from the pipeline.')
            else:
                self.logger.warning(
                    f'{method.__name__} is not in the pipeline.')

    def empty_pipeline(self) -> None:
        """
        Removes all methods from the pipeline.
        """
        if self.pipeline:
            self.pipeline = []
            self.logger.info(
                "All methods have been removed from the pipeline.")
        else:
            self.logger.info("The pipeline is already empty.")

    def reprioritize_pipeline_methods(self, method_names: List[str]) -> None:
        """
        Reprioritize the pipeline methods based on the provided method names.
        The methods will be executed in the order specified by the method names list.
        Any methods not included in the list will retain their original order.
        """
        if self.pipeline:
            new_pipeline = []
            existing_methods = set(self.pipeline)

            for method_name in method_names:
                if (method := next((m for m in self.pipeline if m.__name__ == method_name), None)):
                    new_pipeline.append(method)
                    existing_methods.remove(method)

            new_pipeline += list(existing_methods)
            self.pipeline = new_pipeline
            self.logger.info("Pipeline methods reprioritized.")
        else:
            self.logger.info("The pipeline is currently empty.")

    def execute_pipeline(self, input_text_or_list: Union[str, List[str]],
                         errors: str = 'continue', prioritize_strings: bool = False) -> Union[str, List[str]]:
        """
        Executes the pipeline on the given input text.

        Parameters
        ----------
        input_text_or_list : Union[str, List[str]]
            The text or list of texts to be processed by the pipeline.
        errors : str, optional
            The error handling strategy, either 'continue' or 'stop'.
            Defaults to 'continue'.
        prioritize_strings : bool, optional
            If True, prioritizes string-returning methods in the pipeline.
            Defaults to False.

        Returns
        -------
        Union[str, List[str]]
            The processed text or list of processed texts.
        """
        if errors not in ['continue', 'stop']:
            raise ValueError(
                "Invalid errors value. Valid options are 'continue' and 'stop'.")

        if prioritize_strings:
            self._prioritize_string_methods()

        processed_text = input_text_or_list

        for method in self.pipeline:
            try:
                if isinstance(processed_text, list):
                    processed_text = [method(sent) for sent in processed_text]
                else:
                    processed_text = method(processed_text)
            except Exception as e:
                self._log_error(e)
                if errors == 'stop':
                    break

        return processed_text

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
    numbers_to_words = pipeline_methods.numbers_to_words
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


if __name__ == "__main__":
    preprocessor = PreProcessor(default_pipeline=True)
    text = ' Helllo, I am John Doe!!! My EMAIL is john.doe@email.com. ViSIT ouR wEbSite www.johndoe.com '
    result = preprocessor.execute_pipeline(text)
    print(result)
