## source >>> https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

from abc import ABCMeta


def abstract_attribute(funcobj):
    """A decorator indicating abstract attribute.
    Usage:
        class C(metaclass=CustomABCMeta):
            @abstractmethod
            def my_abstract_attribute(self):
                ...
    """
    funcobj.__is_abstract_attribute__ = True
    return funcobj


class CustomABCMeta(ABCMeta):
    """
    Defines a custom metaclass, CustomABCMeta, which extends ABCMeta to support abstract attributes.
    """

    def __call__(cls, *args, **kwargs):
        instance = ABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {name for name in dir(instance) if getattr(getattr(instance, name), "__is_abstract_attribute__", False)}
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with" " abstract attributes: {}".format(cls.__name__, ", ".join(abstract_attributes))
            )
        return instance
