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
        mass: int=0,
        length: int=0,
        time: int=0,
    ):
        self.mass = mass
        self.length = length
        self.time = time

    def __getitem__(self, item: int) -> int:
        return [self.mass, self.length, self.time][item]

    def __iter__(self) -> Iterable[int]:
        for item in [self.mass, self.length, self.time]:
            yield item

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return False

        return all([
            self.mass == other[0],
            self.length == other[1],
            self.time == other[2],
        ])

    def __neg__(self) -> Dim:
        return Dim(-self.mass, -self.length, -self.time)

    def __add__(self, other: object) -> Dim:
        if not isinstance(other, Dim):
            return NotImplemented

        return Dim(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time
        )

    def __sub__(self, other: object) -> Dim:
        if not isinstance(other, Dim):
            return NotImplemented

        return Dim(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time
        )

    def __str__(self) -> str:
        return f'[{self.mass},{self.length},{self.time}]'


class _Qtag(enum.Enum):
    """ Internal tag for physical quantity """
    CST = enum.auto()
    VAR = enum.auto()
    ADD = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()


class Quant:
    """ Class to represent physical quantity """
    by_name: Dict[str, Quant] = {}

    def __init__(
        self,
        name: str,
        operands: List[Quant],
        dim: Tuple[int, int, int],
        tag: _Qtag=_Qtag.VAR,
        value: Fraction=Fraction(0),
    ):
        if name not in Quant.by_name:
            Quant.by_name[name] = self

        self.name = name
        self.operands = operands
        self.dim = Dim(dim[0], dim[1], dim[2])
        self.tag = tag
        self.value = value if self.tag is _Qtag.CST else None

    @classmethod
    def from_name(cls, name: str) -> Quant:
        return Quant.by_name[name]

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quant):
            return NotImplemented

        if self.is_const and other.is_const:
            return self.value == other.value

        if self.is_var and other.is_var:
            return self.name == other.name

        if self.is_sum and other.is_sum or self.is_prod and other.is_prod:
            operands_count = len(self.operands)
            if operands_count != len(other.operands):
                return False

            # This logic is not needed assuming that
            # each `.operands` array is sorted by some rules.
            # Please remove the block after the issue is resolved.
            marked: Set[int] = set()
            for i in range(operands_count):
                failed = True
                for j in range(operands_count):
                    if j in marked:
                        continue
                    if self.operands[i] == other.operands[j]:
                        marked.add(j)
                        failed = False
                        break
                if failed:
                    return False
            return True

        if self.is_frac and other.is_frac:
            fst = self.operands[0] == other.operands[0]
            snd = self.operands[1] == other.operands[1]
            return fst and snd

        return False

    def __copy__(self) -> Quant:
        if self.is_const:
            return Quant.from_const(
                value=self.value,
                dim=(self.dim[0], self.dim[1], self.dim[2]),
            )

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
        return NotImplemented

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
        return NotImplemented

    # other * self
    def __rmul__(self, other: Any) -> Quant:
        return self * other

    # self / other
    def __truediv__(self, other: Any) -> Quant:
        return NotImplemented

    # other / self
    def __rtruediv__(self, other: Any) -> Quant:
        return NotImplemented


def _factor(x: Quant, y: Quant) -> Tuple[Quant, Quant, Quant]:
    return NotImplemented


def _rewrite_add(x: Quant, y: Quant) -> Optional[Quant]:
    return NotImplemented


def _rewrite_mul(x: Quant, y: Quant) -> Optional[Quant]:
    return NotImplemented
