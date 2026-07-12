"""Generic middleware patching helpers for method interception."""

import types
import inspect

class Context:
    """Carry mutable call context through middleware hooks."""

    def __init__(self, instance, args, kwargs):
        """Store target instance and positional/keyword arguments."""
        self.instance = instance
        self.args = args
        self.kwargs = kwargs
    
    def __str__(self):
        return f"{self.args} \n {self.kwargs}"


class Middleware:
    """
    Middleware class for handling function processing.

    This class provides a hook-based middleware pattern with before and after
    processing stages. Subclasses can override these methods to implement
    custom logic for intercepting and modifying function handling.

    Methods:
        before(ctx): Called before the main operation. Receives the context object.
        after(ctx, result): Called after the main operation. Receives the context 
                           object and the result, returning the (possibly modified) result.
    """
    def before(self, ctx):
        pass

    def after(self, ctx, result):
        return result

def build_llm_wrapper(func, middlewares):
    """Build a sync or async wrapper that executes middleware hooks."""
    if inspect.iscoroutinefunction(func):

        async def async_wrapper(self, *args, **kwargs):
            ctx = Context(self, args, kwargs)
            cur_result = None
            for m in middlewares:
                result = m.before(ctx)
                if result is not None:
                    if cur_result is not None:
                        raise RuntimeError("Multiple middlewares returned a result in before hooks, which is not supported.")
                    cur_result = result
            if cur_result is not None:
                result = cur_result
            else:
                result = await func(self, *ctx.args, **ctx.kwargs)

            for m in reversed(middlewares):
                result = m.after(ctx, result)

            return result

        return async_wrapper

    else:

        def sync_wrapper(self, *args, **kwargs):
            ctx = Context(self, args, kwargs)
            cur_result = None
            for m in middlewares:
                result = m.before(ctx)
                if result is not None:
                    if cur_result is not None:
                        raise RuntimeError("Multiple middlewares returned a result in before hooks, which is not supported.")
                    cur_result = result
            if cur_result is not None:
                result = cur_result
            else:
                result = func(self, *ctx.args, **ctx.kwargs)

            for m in reversed(middlewares):
                result = m.after(ctx, result)

            return result

        return sync_wrapper

def patch_with_middlewares(instance, method_name, middlewares):
    """
    Patch a target instance method with middleware processing chain.

    Args:
        instance: The instance whose method will be patched.
        method_name: The method name to be intercepted.
        middlewares: Ordered middleware objects to inject.
    """
    method = getattr(instance, method_name)

    if inspect.ismethod(method):
        func = method.__func__
    else:
        func = method
    
    wrapped = build_llm_wrapper(func, middlewares)
    setattr(instance, method_name, types.MethodType(wrapped, instance))