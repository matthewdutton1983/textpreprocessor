from functools import wraps


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
