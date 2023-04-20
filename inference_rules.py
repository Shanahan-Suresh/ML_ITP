from formal_logic import *

class InferenceRule:
    def __init__(self, name, premises, conclusion):
        self.name = name
        self.premises = premises
        self.conclusion = conclusion

    def __repr__(self):
        premises_str = ', '.join(map(str, self.premises))
        return f"{self.name}: {premises_str} ⊢ {self.conclusion}"

## Propositional Logic Inference Rules ##

# Modus Ponens: (A, A → B) ⊢ B
modus_ponens = InferenceRule("Modus Ponens", [Variable("A"), Implies(Variable("A"), Variable("B"))], Variable("B"))

# Modus Tollens: (¬B, A → B) ⊢ ¬A
modus_tollens = InferenceRule("Modus Tollens", [Not(Variable("B")), Implies(Variable("A"), Variable("B"))], Not(Variable("A")))

# Hypothetical Syllogism: (A → B, B → C) ⊢ (A → C)
hypothetical_syllogism = InferenceRule(
    "Hypothetical Syllogism",
    [Implies(Variable("A"), Variable("B")), Implies(Variable("B"), Variable("C"))],
    Implies(Variable("A"), Variable("C"))
)

# Disjunctive Syllogism: (A ∨ B, ¬A) ⊢ B
disjunctive_syllogism = InferenceRule(
    "Disjunctive Syllogism",
    [Or(Variable("A"), Variable("B")), Not(Variable("A"))],
    Variable("B")
)

# Addition: A ⊢ (A ∨ B)
addition = InferenceRule(
    "Addition",
    [Variable("A")],
    Or(Variable("A"), Variable("B"))
)

# Simplification: (A ∧ B) ⊢ A
simplification = InferenceRule(
    "Simplification",
    [And(Variable("A"), Variable("B"))],
    Variable("A")
)

# Resolution: (A ∨ B, ¬A ∨ C) ⊢ (B ∨ C)
resolution = InferenceRule(
    "Resolution",
    [Or(Variable("A"), Variable("B")), Or(Not(Variable("A")), Variable("C"))],
    Or(Variable("B"), Variable("C"))
)

#Arithmetic Inference Rules

# Associativity: (a + b) + c = a + (b + c), (a * b) * c = a * (b * c)
associativity_add = InferenceRule(
    "Associativity (Addition)",
    [],
    Eq(Add(Add(Variable("a"), Variable("b")), Variable("c")), Add(Variable("a"), Add(Variable("b"), Variable("c"))))
)

associativity_mul = InferenceRule(
    "Associativity (Multiplication)",
    [],
    Eq(Mul(Mul(Variable("a"), Variable("b")), Variable("c")), Mul(Variable("a"), Mul(Variable("b"), Variable("c"))))
)

# Commutativity: a + b = b + a, a * b = b * a
commutativity_add = InferenceRule(
    "Commutativity (Addition)",
    [],
    Eq(Add(Variable("a"), Variable("b")), Add(Variable("b"), Variable("a")))
)

commutativity_mul = InferenceRule(
    "Commutativity (Multiplication)",
    [],
    Eq(Mul(Variable("a"), Variable("b")), Mul(Variable("b"), Variable("a")))
)

# Distributivity: a * (b + c) = (a * b) + (a * c)
distributivity = InferenceRule(
    "Distributivity",
    [],
    Eq(Mul(Variable("a"), Add(Variable("b"), Variable("c"))), Add(Mul(Variable("a"), Variable("b")), Mul(Variable("a"), Variable("c"))))
)

# Identity: a + 0 = a, a * 1 = a
identity_add = InferenceRule(
    "Identity (Addition)",
    [],
    Eq(Add(Variable("a"), Number(0)), Variable("a"))
)

identity_mul = InferenceRule(
    "Identity (Multiplication)",
    [],
    Eq(Mul(Variable("a"), Number(1)), Variable("a"))
)

# Inverse: a + (-a) = 0, a * (1/a) = 1 (if a ≠ 0)
inverse_add = InferenceRule(
    "Inverse (Addition)",
    [],
    Eq(Add(Variable("a"), UnaryOp(Connective.SUB, Variable("a"))), Number(0))
)

inverse_mul = InferenceRule(
    "Inverse (Multiplication)",
    [Not(Eq(Variable("a"), Number(0)))],
    Eq(Mul(Variable("a"), Div(Number(1), Variable("a"))), Number(1))
)

# Methods to apply inference rules

# Associativity Add
def apply_associativity_add(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.ADD:
        if isinstance(expr.left, BinaryOp) and expr.left.op == Connective.ADD:
            # Left-associative to right-associative
            a = expr.left.left
            b = expr.left.right
            c = expr.right
            return BinaryOp(a, Connective.ADD, BinaryOp(b, Connective.ADD, c))
        elif isinstance(expr.right, BinaryOp) and expr.right.op == Connective.ADD:
            # Right-associative to left-associative
            a = expr.left
            b = expr.right.left
            c = expr.right.right
            return BinaryOp(BinaryOp(a, Connective.ADD, b), Connective.ADD, c)
    return expr

# Associativity Mul
def apply_associativity_mul(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.MUL:
        if isinstance(expr.left, BinaryOp) and expr.left.op == Connective.MUL:
            # Left-associative to right-associative
            a = expr.left.left
            b = expr.left.right
            c = expr.right
            return BinaryOp(a, Connective.MUL, BinaryOp(b, Connective.MUL, c))
        elif isinstance(expr.right, BinaryOp) and expr.right.op == Connective.MUL:
            # Right-associative to left-associative
            a = expr.left
            b = expr.right.left
            c = expr.right.right
            return BinaryOp(BinaryOp(a, Connective.MUL, b), Connective.MUL, c)
    return expr

# Commutative Add
def apply_commutativity_add(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.ADD:
        return BinaryOp(expr.right, Connective.ADD, expr.left)
    return expr

# Commutative Mul
def apply_commutativity_mul(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.MUL:
        return BinaryOp(expr.right, Connective.MUL, expr.left)
    return expr

# Distributivity
def apply_distributivity(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.MUL:
        if isinstance(expr.right, BinaryOp) and expr.right.op == Connective.ADD:
            # Distribute left multiplication
            a = expr.left
            b = expr.right.left
            c = expr.right.right
            return Add(Mul(a, b), Mul(a, c))
        elif isinstance(expr.left, BinaryOp) and expr.left.op == Connective.ADD:
            # Distribute right multiplication
            a = expr.left.left
            b = expr.left.right
            c = expr.right
            return Add(Mul(a, c), Mul(b, c))
    return expr

# Identity Add
def apply_identity_add(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.ADD:
        if isinstance(expr.left, Number) and expr.left.value == 0:
            return expr.right
        elif isinstance(expr.right, Number) and expr.right.value == 0:
            return expr.left
    return expr

# Identity Mul
def apply_identity_mul(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.MUL:
        if isinstance(expr.left, Number) and expr.left.value == 1:
            return expr.right
        elif isinstance(expr.right, Number) and expr.right.value == 1:
            return expr.left
    return expr

# Inverse Add
def apply_inverse_add(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.ADD:
        if isinstance(expr.left, UnaryOp) and expr.left.op == Connective.NEG and expr.left.expr == expr.right:
            return Number(0)
        elif isinstance(expr.right, UnaryOp) and expr.right.op == Connective.NEG and expr.right.expr == expr.left:
            return Number(0)
    return expr

# Inverse Mul
def apply_inverse_mul(expr):
    if isinstance(expr, BinaryOp) and expr.op == Connective.MUL:
        if isinstance(expr.left, Fraction) and expr.left.denominator == expr.right:
            return Number(1)
        elif isinstance(expr.right, Fraction) and expr.right.denominator == expr.left:
            return Number(1)
    return expr



#print(modus_ponens)  # Output: Modus Ponens: A, (A → B) ⊢ B
#print(modus_tollens)  # Output: Modus Tollens: (¬B), (A → B) ⊢ (¬A)

