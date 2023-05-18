# Import standard libraries
import inspect
import logging
from functools import wraps
from typing import Union, List, Callable


logging.basicConfig(level=logging.INFO)


class Pipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.pipeline = []

    def _log_error(self, e: Exception) -> None:
        function_name = inspect.currentframe().f_back.f_code.co_name
        self.logger.error(
            f'Error occurred in function {function_name}: {str(e)}')

    def pipeline_method(func):
        """Decorator to mark methods that can be added to the pipeline"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        wrapper.is_pipeline_method = True
        return wrapper
    
    def add_methods(self, methods: Union[List[Callable], Callable]) -> None:
        """Add a method to the pipeline"""
        if not isinstance(methods, list):
            methods = [methods]

        for method in methods:
            if getattr(method.__func__, 'is_pipeline_method', False):
                if method not in self.pipeline:
                    self.pipeline.append(method)
                    self.logger.info(f'{method.__name__} has been added to the pipeline.')
                else:
                    self.logger.warning(f'{method.__name__} is already in the pipeline.')
            else:
                self.logger.error(f'{method.__name__} is not a valid pipeline method.')

    def remove_methods(self, methods: Union[List[Callable], Callable]) -> None:
        """Remove a method from the pipeline"""
        if not isinstance(methods, list):
            methods = [methods]
          
        for method in methods:
            if method in self.pipeline:
                self.pipeline.remove(method)
                self.logger.info(f'{method.__name__} has been removed from the pipeline.')
            else:
                self.logger.error(f'{method.__name__} is not in the pipeline.')

    def view_pipeline(self) -> None:
        """Print the name of each method in the pipeline"""
        if self.pipeline:
            self.logger.info("Current pipeline configuration:")
            for i, method in enumerate(self.pipeline):
                self.logger.info(f'{i+1}: {method.__name__}')
        else:
            self.logger.info('The pipeline is currently empty.')

    def _prioritize_string_methods(self) -> None:
        """Gives priority to string methods"""
        str_methods: List[Callable] = []
        other_methods: List[Callable] = []

        for method in self.pipeline:
            if getattr(method.__func__, 'is_pipeline_method', False):
                if method.__annotations__.get('return') == str:
                    str_methods.append(method)
                else:
                    other_methods.append(method)
        
        self.pipeline = str_methods + other_methods

    def execute_pipeline(self, input_text: str, errors: str = 'continue') -> str:
        # Update this function to provide flexibility to override prioritization
        # Update so user has to pass in the pipeline; no need for separate execute default method then
        if errors not in ['continue', 'stop']:
            raise ValueError("Invalid errors value. Valid options are 'continue' and 'stop'.")

        self._prioritize_string_methods()

        for method in self.pipeline:
            try:
                processed_text = method(input_text)
            except Exception as e:
                self._log_error(e)
                if errors == 'stop':
                    break
        
        return processed_text
    
    def clear_pipeline(self) -> None:
        if self.pipeline:
            self.pipeline = []
            self.logger.info("All methods have been removed from the pipeline.")
        else:
            self.logger.info("The pipeline is already empty.")

