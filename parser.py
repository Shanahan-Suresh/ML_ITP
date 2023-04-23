from lark import Lark, Transformer, v_args
from formal_logic import *
from proof_engine import dfs_search_inference_rules

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

# Propositional logic operations

def and_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = And(a, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def or_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = Or(a, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def not_operation(a: Expression) -> Expression:
    premises = {a}
    conclusion = Not(a)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def implies_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = Implies(a, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def biconditional_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = Biconditional(a, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def xor_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = Xor(a, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def add_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = BinaryOp(a, Connective.ADD, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def sub_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = BinaryOp(a, Connective.SUB, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def mul_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = BinaryOp(a, Connective.MUL, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def div_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = BinaryOp(a, Connective.DIV, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def eq_operation(a: Expression, b: Expression) -> Expression:
    premises = {a, b}
    conclusion = BinaryOp(a, Connective.EQ, b)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None

def neg_operation(a: Expression) -> Expression:
    premises = {a}
    conclusion = UnaryOp(Connective.NEG, a)
    applied_rules, _ = dfs_search_inference_rules(premises, conclusion)
    return conclusion if applied_rules else None




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
    parser = Lark(grammar, start="start", parser="lalr", transformer=ExprTransformer())
    try:
        return parser.parse(expr)
    except Exception as e:
        print(f"Error parsing expression: {e}")

def process_expression(expr: Expression):
    if isinstance(expr, BinaryOp):
        # Process binary operation
        process_expression(expr.left)
        process_expression(expr.right)
        # Perform the operation based on the connective
        if expr.op == Connective.AND:
            and_operation(expr.left, expr.right)
        elif expr.op == Connective.OR:
            or_operation(expr.left, expr.right)
        elif expr.op == Connective.IMPLIES:
            implies_operation(expr.left, expr.right)
        elif expr.op == Connective.BICONDITIONAL:
            biconditional_operation(expr.left, expr.right)
        elif expr.op == Connective.XOR:
            xor_operation(expr.left, expr.right)
        elif expr.op == Connective.ADD:
            add_operation(expr.left, expr.right)
        elif expr.op == Connective.SUB:
            sub_operation(expr.left, expr.right)
        elif expr.op == Connective.MUL:
            mul_operation(expr.left, expr.right)
        elif expr.op == Connective.DIV:
            div_operation(expr.left, expr.right)
        elif expr.op == Connective.EQ:
            eq_operation(expr.left, expr.right)
        else:
            raise ValueError(f"Unsupported binary connective: {expr.op}")

    elif isinstance(expr, UnaryOp):
        # Process unary operation
        process_expression(expr.expr)
        # Perform the operation based on the connective
        if expr.op == Connective.NOT:
            not_operation(expr.expr)
        elif expr.op == Connective.NEG:
            neg_operation(expr.expr)
        else:
            raise ValueError(f"Unsupported unary connective: {expr.op}")

    elif isinstance(expr, Variable) or isinstance(expr, Number) or isinstance(expr, NumericalVariable) or isinstance(expr, Fraction):
        # Handle variables, numbers, numerical variables, and fractions
        # This could involve storing them in a dictionary, for example
        pass
    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")

# Create variables
A = Variable("A")
B = Variable("B")

# Test the and_operation function
result = and_operation(A, B)
print(result)  # Output: (A ∧ B)
