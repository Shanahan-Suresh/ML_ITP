from lark import Lark, Transformer, v_args
from formal_logic import *

grammar = r"""
    ?start: expr
    ?expr: binary_op | unary_op | parens | CNAME | NUMBER
    parens: "(" expr ")"
    binary_op: expr BINARY_CONNECTIVE expr
    unary_op: UNARY_CONNECTIVE expr
    BINARY_CONNECTIVE: "∧" | "∨" | "→" | "↔" | "⊕" | "+" | "−" | "*" | "/" | "="
    UNARY_CONNECTIVE: "¬" | "-"
    %import common.CNAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""


class TreeToExpression(Transformer):
    @v_args(inline=True)
    def CNAME(self, name):
        return Variable(name)

    @v_args(inline=True)
    def NUMBER(self, value):
        return Number(float(value))

    def binary_op(self, items):
        left, op, right = items
        return BinaryOp(left, Connective[op], right)

    def unary_op(self, items):
        op, expr = items
        return UnaryOp(Connective[op], expr)

    def start(self, items):
        return items[0]

    def expr(self, items):
        return items[0]

class ExprTransformer(Transformer):
    def parens(self, args):
        return args[0]

    def binary_op(self, args):
        left, op, right = args
        return BinaryOp(left, CONNECTIVE_SYMBOLS.inv[op], right)

    def unary_op(self, args):
        op, expr = args
        return UnaryOp(CONNECTIVE_SYMBOLS.inv[op], expr)

def parse_expression(expr: str) -> Expression:
    expr = expr.replace("implies", "→").replace("and", "∧").replace("or", "∨").replace("not", "¬").replace("biconditional", "↔").replace("xor", "⊕")
    parser = Lark(grammar, start="start", parser="lalr", transformer=ExprTransformer())
    try:
        return parser.parse(expr)
    except Exception as e:
        print(f"Error parsing expression: {e}")

