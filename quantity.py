""" Quantity """
from __future__ import annotations

import enum
from functools import reduce
from fractions import Fraction
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple


class Dim:
    """ Class to represent dimension of physical quantity """
    def __init__(
        self,
        length: int=0,
        mass: int=0,
        time: int=0,
    ):
        self.length = length
        self.mass = mass
        self.time = time

    def __getitem__(self, item: int) -> int:
        return [self.length, self.mass, self.time][item]

    def __iter__(self) -> Iterable[int]:
        for item in [self.length, self.mass, self.time]:
            yield item

    def __len__(self) -> int:
        return 3

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return False

        return all([
            self.length == other[0],
            self.mass == other[1],
            self.time == other[2],
        ])

    def __neg__(self) -> Dim:
        return Dim(-self.length, -self.mass, -self.time)

    def __add__(self, other: Any) -> Dim:
        return Dim(
            self.length + other[0],
            self.mass + other[1],
            self.time + other[2],
        )

    def __sub__(self, other: Any) -> Dim:
        return Dim(
            self.length - other[0],
            self.mass - other[1],
            self.time - other[2],
        )

    def __str__(self) -> str:
        return f'({self.length},{self.mass},{self.time})'


class _Qtag(enum.Enum):
    """ Internal tag for physical quantity """
    CST = enum.auto()
    VAR = enum.auto()
    ADD = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()


class Quant:
    """ Class to represent physical quantity """
    def __init__(
        self,
        name: str,
        operands: List[Quant],
        dim: Tuple[int, int, int],
        tag: _Qtag=_Qtag.VAR,
        value: Fraction=Fraction(0),
    ):
        self.name = name
        self.operands = operands
        self.dim = Dim(dim[0], dim[1], dim[2])
        self.tag = tag
        self.value = value if self.tag is _Qtag.CST else None

    @classmethod
    def length(cls, name: str) -> Quant:
        return Quant(
            name=name,
            operands=[],
            dim=(1, 0, 0),
        )

    @classmethod
    def mass(cls, name: str) -> Quant:
        return Quant(
            name=name,
            operands=[],
            dim=(0, 1, 0),
        )

    @classmethod
    def time(cls, name: str) -> Quant:
        return Quant(
            name=name,
            operands=[],
            dim=(0, 0, 1),
        )

    @classmethod
    def from_const(
        cls,
        value: Any,
        dim: Tuple[int, int, int]=(0,0,0),
    ) -> Quant:
        return Quant(
            name='__constant__',
            operands=[],
            dim=dim,
            tag=_Qtag.CST,
            value=Fraction(value),
        )

    @property
    def is_const(self) -> bool:
        return self.tag is _Qtag.CST

    @property
    def is_var(self) -> bool:
        return self.tag is _Qtag.VAR

    @property
    def is_sum(self) -> bool:
        return self.tag is _Qtag.ADD

    @property
    def is_prod(self) -> bool:
        return self.tag is _Qtag.MUL

    @property
    def is_frac(self) -> bool:
        return self.tag is _Qtag.DIV

    def __str__(self) -> str:
        if self.is_const:
            return str(self.value)

        if self.is_var:
            return self.name

        op = '+' if self.is_sum else '*' if self.is_prod else '/'
        return '(' + op.join([str(x) for x in self.operands]) + ')'

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Quant):
            return self.value == other

        if self.is_const and other.is_const:
            return self.value == other.value

        if self.is_var and other.is_var:
            return self.name == other.name

        if self.is_sum and other.is_sum or self.is_prod and other.is_prod:
            return all(x == y for x, y in zip(self.operands, other.operands))

        if self.is_frac and other.is_frac:
            fst = self.operands[0] == other.operands[0]
            snd = self.operands[1] == other.operands[1]
            return fst and snd

        return False

    def __copy__(self) -> Quant:
        if self.is_const:
            dim =  (self.dim[0], self.dim[1], self.dim[2])
            return Quant.from_const(value=self.value, dim=dim)

        if self.is_var:
            return Quant(
                name=self.name,
                operands=[],
                dim=(self.dim[0], self.dim[1], self.dim[2]),
                tag=_Qtag.VAR,
            )

        copied_operands = [x.__copy__() for x in self.operands]
        return Quant(
            name=self.name,
            operands=copied_operands,
            dim=(self.dim[0], self.dim[1], self.dim[2]),
            tag=self.tag,
        )

    # +self
    def __pos__(self) -> Quant:
        return self

    # -self
    def __neg__(self) -> Quant:
        return -1 * self

    # self + other
    def __add__(self, other: Any) -> Quant:
        assert other is not None

        dim = (self.dim[0], self.dim[1], self.dim[2])

        if not isinstance(other, Quant):
            other = Quant.from_const(other, dim)

        assert self.dim == other.dim

        # gather terms
        def flatten(xs):
            while any(x.is_sum for x in xs):
                xs = sum((x.operands if x.is_sum else [x] for x in xs), [])
            return xs

        xs = flatten([self, other])

        # reduce terms
        ys: List[Quant] = []
        for x in xs:
            found = None
            for i, y in enumerate(ys):
                found = _rewrite_add(x, y)
                if found is not None:
                    ys[i] = found
                    break
            if found is None:
                ys.append(x)

        # remove terms
        terms = [y for y in ys
                 if not (y.is_const and y.value == 0)]
        terms.sort(key=lambda term: term.name)

        if len(terms) == 0:
            return Quant.from_const(0, dim)
        elif len(terms) == 1:
            return terms[0]

        return Quant(
            name='__sum__',
            operands=terms,
            dim=dim,
            tag=_Qtag.ADD,
        )

    # other + self
    def __radd__(self, other: Any) -> Quant:
        return self + other

    # self - other
    def __sub__(self, other: Any) -> Quant:
        return self + (-other)

    # other - sub
    def __rsub__(self, other: Any) -> Quant:
        return -self + other

    # self * other
    def __mul__(self, other: Any) -> Quant:
        assert other is not None

        if not isinstance(other, Quant):
            other = Quant.from_const(other)

        dim = (self.dim[0] + other.dim[0],
               self.dim[1] + other.dim[1],
               self.dim[2] + other.dim[2])

        # gather terms
        def flatten(xs):
            while any(x.is_prod for x in xs):
                xs = sum((x.operands if x.is_prod else [x] for x in xs), [])
            return xs

        xs = flatten([self, other])

        # reduce terms
        ys: List[Quant] = []
        for x in xs:
            found = None
            for i, y in enumerate(ys):
                found = _rewrite_mul(x, y)
                if found is not None:
                    ys[i] = found
                    break
            if found is None:
                ys.append(x)

        # remove terms
        terms = [y for y in ys
                 if not (y.is_const and y.value == 1 and y.dim == (0, 0, 0))]
        terms.sort(key=lambda term: term.name)

        if len(terms) == 0:
            return Quant.from_const(1, dim)
        elif len(terms) == 1:
            return terms[0]

        return Quant(
            name='__mul__',
            operands=terms,
            dim=dim,
            tag=_Qtag.MUL,
        )

    # other * self
    def __rmul__(self, other: Any) -> Quant:
        return self * other

    # self / other
    def __truediv__(self, other: Any) -> Quant:
        assert other is not None

        if not isinstance(other, Quant):
            other = Quant.from_const(other)

        dim = (self.dim[0] - other.dim[0],
               self.dim[1] - other.dim[1],
               self.dim[2] - other.dim[2])

        # reduce terms
        _, numer, denom = _factor(self, other)

        # remove terms
        if numer.is_const and numer.value == 0:
            return Quant.from_const(0, dim)
        elif denom.is_const and denom.value is not None:
            inv_dim = (-other.dim[0], -other.dim[1], -other.dim[2])
            return numer * Quant.from_const(1 / denom.value, inv_dim)

        return Quant(
            name='__div__',
            operands=[numer, denom],
            dim=dim,
            tag=_Qtag.DIV,
        )

    # other / self
    def __rtruediv__(self, other: Any) -> Quant:
        assert other is not None

        return Quant.from_const(other) / self


def _factor(x: Quant, y: Quant) -> Tuple[Quant, Quant, Quant]:
    """ factor two quantities and find greatest common division """
    assert x is not None and y is not None

    one = Quant.from_const(1)

    return one, x, y


def _rewrite_add(x: Quant, y: Quant) -> Optional[Quant]:
    """ rewrite sum form """
    assert x is not None and y is not None

    return None


def _rewrite_mul(x: Quant, y: Quant) -> Optional[Quant]:
    """ rewrite product form """
    assert x is not None and y is not None

    return None
