""" Test of Quantity """
from __future__ import annotations

from pytest import fixture
from quantity import Quant as Q

###############################################################################
### Test context                                                            ###
###############################################################################
@fixture
def l():
    yield Q.length('l')

@fixture
def m():
    yield Q.mass('m')

@fixture
def t():
    yield Q.time('t')

@fixture
def zero():
    yield Q.from_const(0)

@fixture
def one():
    yield Q.from_const(1)

###############################################################################
### Test functions                                                          ###
###############################################################################
def test_vardecl(l, m, t):
    # test variable name
    assert l.name == 'l'
    assert m.name == 'm'
    assert t.name == 't'
    # test dimension
    assert l.dim == (1, 0, 0)
    assert m.dim == (0, 1, 0)
    assert t.dim == (0, 0, 1)
    # test operands are empty
    assert len(l.operands) == 0
    assert len(m.operands) == 0
    assert len(t.operands) == 0
    # test value is none
    assert l.value is None
    assert m.value is None
    assert t.value is None

def test_const(zero, one):
    # test constant value
    assert zero.value == 0
    assert one.value == 1
    # test constant variable name
    assert zero.name == one.name
    # test operands are empty
    assert len(zero.operands) == 0
    assert len(one.operands) == 0
