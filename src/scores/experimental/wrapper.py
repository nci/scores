"""
API Wrapper around an underlying module, in this case `scores`.

Allows manipulation of data prior to calling base `scores` function.
"""


from typing import Any, Callable, Union
import scores

class APIWrapper:
    """
    Base api wrapper of `scores` for use with other frameworks
    """
    def __init__(self, function: Callable = scores):
        """Base wrapper for api control of `scores`

        Args:
            function (Callable, optional): 
                Function to wrap. Provides access to underlying attributes.
                Defaults to `scores`.
        """        
        if function is None:
            function = scores
        self.function = function

    def help(self):
        """Get help for underlying function"""
        return help(self.function)
    
    def __getattr__(self, key: str) -> Union[Callable, Any]:
        """Get underlying attribute from self.function

        Args:
            key (str): 
                Attribute name to find

        Raises:
            AttributeError: 
                If function has no attribute key

        Returns:
            (Union[Callable, Any]): 
                Underlying attribute
        """        
        if key == "function":
            raise AttributeError(f"{self} has no attribute {key!r}")
        
        if not hasattr(self.function, key):
            raise AttributeError(f"{self.function} has no attribute {key!r}")
        
        new_func = getattr(self.function, key)
        
        return self.__class__(new_func)
    
    def __dir__(self):
        """Get `__dir__` of underlying function"""
        return self.function.__dir__()
    
    def __repr__(self):
        """repr"""
        return repr(self.function)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call underlying function
        
        Raises:
            TypeError: 
                If function cannot be called
        """
        if not hasattr(self.function, '__call__'):
            raise TypeError(f"{type(self.function)}: {self.function} can not be called.")
        return self.function(*args, **kwargs)
    
    def callback(self, **callback_kwargs):
        """
        Create a callback function where any passed `callback_kwargs` are also passed.

        Args:
            **callback_kwargs (Any): 
                Extra kwargs to pass when callbacked
        """     
        callback_kwargs: dict = callback_kwargs   

        def callback_decorator(*args, **kwargs):
            callback_kwargs.update(kwargs)
            return self.__call__(*args, **callback_kwargs)
        
        return callback_decorator
