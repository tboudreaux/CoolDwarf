from typing import Protocol, TypeVar, Union

class SupportsArithmetic(Protocol):
    def __add__(self, other: Union[int, float, 'SupportsArithmetic']) -> 'SupportsArithmetic': ...
    def __sub__(self, other: Union[int, float, 'SupportsArithmetic']) -> 'SupportsArithmetic': ...
    def __mul__(self, other: Union[int, float, 'SupportsArithmetic']) -> 'SupportsArithmetic': ...
    def __truediv__(self, other: Union[int, float, 'SupportsArithmetic']) -> 'SupportsArithmetic': ...

Arithmetic = TypeVar('Arithmetic', bound=SupportsArithmetic)