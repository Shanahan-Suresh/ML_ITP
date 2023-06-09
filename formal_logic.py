﻿
from abc import abstractmethod
from enum import Enum, auto
from bidict import bidict

## Abstract language syntex
class Connective(Enum):

    # Propositional logic
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    BICONDITIONAL = auto()
    XOR = auto()

    # arithmetic operations
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    EQ = auto()
    NEG = auto()

#dictionary for mapping symbols
CONNECTIVE_SYMBOLS = bidict({
    Connective.AND: "∧",
    Connective.OR: "∨",
    Connective.NOT: "¬",
    Connective.IMPLIES: "→",
    Connective.BICONDITIONAL: "↔",
    Connective.XOR: "⊕",
    Connective.ADD: "+",
    Connective.SUB: "−",
    Connective.MUL: "*",
    Connective.DIV: "/",
    Connective.EQ: "=",
    Connective.NEG: "-"
})


class Expression:
    @abstractmethod
    def __eq__(self, other):
        pass    

class Variable(Expression):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.name == other.name


class Number(Expression):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Number):
            return False
        return self.value == other.value

class Fraction(Expression):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self):
        return f"({self.numerator}/{self.denominator})"

    def __hash__(self):
        return hash((self.numerator, self.denominator))

    def __eq__(self, other):
        return isinstance(other, Fraction) and self.numerator == other.numerator and self.denominator == other.denominator

class NumericalVariable(Expression):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, NumericalVariable):
            return False
        return self.name == other.name

class BinaryOp(Expression):
    # left and right operands, op refers to applied operator
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op.name} {self.right})"

    def __eq__(self, other):
        if not isinstance(other, BinaryOp):
            return False
        return self.left == other.left and self.op == other.op and self.right == other.right

    def __hash__(self):
        return hash((self.left, self.op, self.right))

    def __repr__(self):
        return f"({self.left} {CONNECTIVE_SYMBOLS[self.op]} {self.right})"


class UnaryOp(Expression):
    # single operand and operator
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

    def __repr__(self):
        return f"({self.op.name} {self.expr})"

    def __str__(self):
        if self.op == Connective.NOT:
            return f"({CONNECTIVE_SYMBOLS[self.op]}{self.expr})"
        elif self.op == Connective.NEG:
            return f"({CONNECTIVE_SYMBOLS[self.op]}{self.expr})"
        else:
            raise ValueError(f"Unsupported unary connective: {self.op}")

    def __eq__(self, other):
        if not isinstance(other, UnaryOp):
            return False
        return self.op == other.op and self.expr == other.expr

    def __hash__(self):
        return hash((self.op, self.expr))

    def __repr__(self):
        return f"({CONNECTIVE_SYMBOLS[self.op]}{self.expr})"


# Helper functions - these functions are used to construct expressions

## Propositional logic helpers
def And(a, b):
    return BinaryOp(a, Connective.AND, b)

def Or(a, b):
    return BinaryOp(a, Connective.OR, b)

def Not(a):
    return UnaryOp(Connective.NOT, a)

def Implies(a, b): #conditional relation between two esxpressions, antecedent implies consequent
    return BinaryOp(a, Connective.IMPLIES, b)

def Biconditional(a, b):
    return BinaryOp(a, Connective.BICONDITIONAL, b)

def Xor(a, b):
    return BinaryOp(a, Connective.XOR, b)

## Numerical arithmetic helpers
def Add(a, b):
    return BinaryOp(a, Connective.ADD, b)

def Sub(a, b):
    return BinaryOp(a, Connective.SUB, b)

def Mul(a, b):
    return BinaryOp(a, Connective.MUL, b)

def Div(a, b):
    return BinaryOp(a, Connective.DIV, b)

def Eq(a, b):
    return BinaryOp(a, Connective.EQ, b)

def Neg(a):
    return UnaryOp(Connective.NEG, a)


'''
# TEST SECTION

A = Variable("A")
B = Variable("B")
C = Variable("C")

expr = Implies(And(A, B), Or(Not(A), C))
print(expr)  # Output: ((A AND B) IMPLIES ((NOT A) OR C))

expr2 = Biconditional(A, B)
#print(expr2)

x = NumericalVariable("x")
expr1 = Mul(Number(2), Add(x, Number(7)))
expr2 = Mul(Number(2), Add(x, Number(7)))

equality = Eq(expr1, expr2)
print(equality)  # Output: ((2 * (x + 7)) EQ (2 * (x + 7)))

print(expr1 == expr2)
'''