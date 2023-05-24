# Import standard libraries
import inspect
import logging
from typing import Callable, Optional, Union, List


logging.basicConfig(level=logging.INFO)


class Pipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.pipeline = []

    @property
    def total_methods(self) -> int:
        """Return the number of methds in the pipeline"""
        return len(self.pipeline)

    @property
    def list_methods(self) -> List[Callable]:
        """Return a list of methods in the pipeline"""
        return self.pipeline

    @property
    def is_empty(self) -> bool:
        """Check if the pipeline is empty"""
        return len(self.pipeline) == 0

    @property
    def last_method(self) -> Optional[Callable]:
        """Return the last method added to the pipeline"""
        if self.pipeline:
            return self.pipeline[-1]
        else:
            return None

    @property
    def first_method(self) -> Optional[Callable]:
        """Return the first method added to the pipeline"""
        if self.pipeline:
            return self.pipeline[0]
        else:
            return None

    def _log_error(self, e: Exception) -> None:
        function_name = inspect.currentframe().f_back.f_code.co_name
        self.logger.error(
            f'Error occurred in function {function_name}: {str(e)}')

    def add_methods(self, methods: Union[List[Callable], Callable]) -> None:
        """Add a method to the pipeline"""
        if not isinstance(methods, list):
            methods = [methods]

        for method in methods:
            if getattr(method, 'is_pipeline_method', False):
                if method not in self.pipeline:
                    self.pipeline.append(method)
                    self.logger.info(
                        f'{method.__name__} has been added to the pipeline.')
                else:
                    self.logger.warning(
                        f'{method.__name__} is already in the pipeline.')
            else:
                self.logger.error(
                    f'{method.__name__} is not a valid pipeline method.')

    def remove_methods(self, methods: Union[List[Callable], Callable]) -> None:
        """Remove a method from the pipeline"""
        if not isinstance(methods, list):
            methods = [methods]

        for method in methods:
            if method in self.pipeline:
                self.pipeline.remove(method)
                self.logger.info(
                    f'{method.__name__} has been removed from the pipeline.')
            else:
                self.logger.error(f'{method.__name__} is not in the pipeline.')

    def _prioritize_string_methods(self) -> None:
        """Gives priority to string methods"""
        str_methods: List[Callable] = []
        other_methods: List[Callable] = []

        for method in self.pipeline:
            if getattr(method, 'is_pipeline_method', False):
                if method.__annotations__.get('return') == str:
                    str_methods.append(method)
                else:
                    other_methods.append(method)

        self.pipeline = str_methods + other_methods

    def _load_default_pipeline(self, default_methods: List[Callable]) -> None:
        """Adds a core set of common methods to the pipeline"""
        if self.pipeline:
            self.clear_pipeline()

        self.add_methods(default_methods)
        self.logger.info("Default pipeline loaded.")

    def clear_pipeline(self) -> None:
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
                method = next(
                    (m for m in self.pipeline if m.__name__ == method_name), None
                )
                if method:
                    new_pipeline.append(method)
                    existing_methods.remove(method)

            for method in self.pipeline:
                if method in existing_methods:
                    new_pipeline.append(method)

            self.pipeline = new_pipeline
            self.logger.info("Pipeline methods reprioritized.")
        else:
            self.logger.info("The pipeline is currently empty.")

    def execute_pipeline(self, input_text: str, errors: str = 'continue', prioritize_strings: bool = True) -> dict:
        if errors not in ['continue', 'stop']:
            raise ValueError(
                "Invalid errors value. Valid options are 'continue' and 'stop'.")

        if prioritize_strings:
            self._prioritize_string_methods()

        processed_text = input_text
        exceptions_list = []

        for method in self.pipeline:
            try:
                processed_text = method(processed_text)
            except Exception as e:
                self._log_error(e)
                exceptions_list.append({"method": method.__name__, "error": str(e)})
                if errors == 'stop':
                    break

        return {
            "processed_text": processed_text, 
            "exceptions_list": exceptions_list
        }
    
