def _DeepflowLazyLoader(func):
    """
    `deepflow.utils.lazy_loader._DeepflowLazyLoader`
    ----------------------------------------------
    
    Decorator to lazily load a property.

    Args:
        func: The function that calculates the property value.

    Returns:
        A property object that calls the decorated function only on first access.
    """
    attr_name = f"_{func.__name__}"  # Create private attribute name

    @property
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return wrapper