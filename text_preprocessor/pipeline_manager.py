# Import standard libraries
import inspect
import logging
from typing import Callable, Union, List

# Import third-party libraries
import nltk
from nltk.corpus import stopwords

# Import project code
from text_preprocessor.pipeline import Pipeline


logging.basicConfig(level=logging.INFO)


class PreProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.pipeline = None
        self.supported_languages = ['en', 'es', 'fr', 'pt', 'de', 'ru', 'ar']
        self.stop_words = set(stopwords.words('english'))

        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)

    def _log_error(self, e: Exception) -> None:
        function_name = inspect.currentframe().f_back.f_code.co_name
        self.logger.error(
            f'Error occurred in function {function_name}: {str(e)}')

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
            self.pipeline.load_default_pipeline(default_methods)

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

    def execute_pipeline(self, input_text: str, errors: str = 'continue') -> str:
        if self.pipeline is None:
            raise ValueError(
                "No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        return self.pipeline.execute_pipeline(input_text, errors)

    def empty_pipeline(self) -> None:
        if self.pipeline is None:
            raise ValueError(
                "No pipeline created. Call create_pipeline() to create a new Pipeline object.")
        self.pipeline.clear_pipeline()
