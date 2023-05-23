def pipeline_method(func):
    """Decorator to mark methods that can be added to the pipeline"""
    func.is_pipeline_method = True
    return func
