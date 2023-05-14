import functools


def patch_callable(callable, condition, type_wrapper):
    @functools.wraps(callable)
    def new_callable(*args, **kwargs):
        result = callable(*args, **kwargs)
        if condition(result):
            return type_wrapper(result)
        return result

    return new_callable
