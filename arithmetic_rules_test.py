from inference_rules import *

def test_print_message(expr1, expr2):
    print('First expr: {}'.format(expr1))
    print('Second expr: {}'.format(expr2))

# Associativity Add Test
def test_AA():
    expr2 = Add(Add(Variable("a"), Variable("b")), Variable("c"))
    expr1 = Add(Variable("a"), Add(Variable("b"), Variable("c")))

    test_print_message(expr1, expr2)
    result = apply_associativity_add(expr1)
    print(result)  # Output: (a + b) + (c + d)
    assert result == expr2

    result2 = apply_associativity_add(result)
    print(result2)  # Output: (a + b) + (c + d)
    assert result2 == expr1

# Associativity Mul Test
def test_AM():
    expr2 = Mul(Mul(Variable("a"), Variable("b")), Variable("c"))
    expr1 = Mul(Variable("a"), Mul(Variable("b"), Variable("c")))

    test_print_message(expr1, expr2)
    result = apply_associativity_mul(expr1)
    print(result)  # Output: (a + b) + (c + d)
    assert result == expr2

    result2 = apply_associativity_mul(result)
    print(result2)  # Output: (a + b) + (c + d)
    assert result2 == expr1

# Commutative Add Test
def test_CA():
    expr1 = Add(Variable("a"), Variable("b"))
    expr2 = Add(Variable("b"), Variable("a"))

    test_print_message(expr1, expr2)
    result = apply_commutativity_add(expr1)
    print(result)  # Output: b + a
    assert result == expr2

# Commutative Mul Test
def test_CM():
    expr1 = Mul(Variable("a"), Variable("b"))
    expr2 = Mul(Variable("b"), Variable("a"))

    test_print_message(expr1, expr2)
    result = apply_commutativity_mul(expr1)
    print(result)  # Output: b + a
    assert result == expr2

# Distibutivity Test
def test_D():
    expr1 = Mul(Variable("a"), Add(Variable("b"), Variable("c")))
    expr2 = Add(Mul(Variable("a"), Variable("b")), Mul(Variable("a"), Variable("c")))

    test_print_message(expr1, expr2)
    result1 = apply_distributivity(expr1)
    print(result1)  # Output: (a * b) + (a * c)
    assert result1 == expr2

    expr1 = Mul(Add(Variable("a"), Variable("b")), Variable("c"))
    expr2 = Add(Mul(Variable("a"), Variable("c")), Mul(Variable("b"), Variable("c")))

    test_print_message(expr1, expr2)
    result2 = apply_distributivity(expr1)
    print(result2)  # Output: (a * c) + (b * c)
    assert result2 == expr2


# Identity Add Test
def test_ID():
    expr1 = Add(Number(0), Variable("a"))
    expr2 = Variable("a")

    test_print_message(expr1, expr2)
    result1 = apply_identity_add(expr1)
    print(result1)  # Output: a
    assert result1 == expr2

    expr1 = Add(Variable("a"), Number(0))
    expr2 = Variable("a")

    test_print_message(expr1, expr2)
    result2 = apply_identity_add(expr1)
    print(result2)  # Output: a
    assert result2 == expr2

# Identity Mul Test
def test_IM():
    expr1 = Mul(Number(1), Variable("a"))
    expr2 = Variable("a")

    test_print_message(expr1, expr2)
    result1 = apply_identity_mul(expr1)
    print(result1)  # Output: a
    assert result1 == expr2

    expr1 = Mul(Variable("a"), Number(1))
    expr2 = Variable("a")

    test_print_message(expr1, expr2)
    result2 = apply_identity_mul(expr1)
    print(result2)  # Output: a
    assert result2 == expr2

# Inverse Add Test
def test_IvA():
    expr1 = Add(UnaryOp(Connective.NEG, Variable("a")), Variable("a"))
    expr2 = Number(0)

    test_print_message(expr1, expr2)
    result1 = apply_inverse_add(expr1)
    print(result1)  # Output: 0
    assert result1 == expr2

    expr1 = Add(Variable("a"), UnaryOp(Connective.NEG, Variable("a")))
    expr2 = Number(0)

    test_print_message(expr1, expr2)
    result2 = apply_inverse_add(expr1)
    print(result2)  # Output: 0
    assert result2 == expr2

# Inverse Add Test
def test_IvM():
    expr1 = Mul(Fraction(1, Variable("a")), Variable("a"))
    expr2 = Number(1)

    test_print_message(expr1, expr2)
    result1 = apply_inverse_mul(expr1)
    print(result1)  # Output: 1
    assert result1 == expr2

    expr1 = Mul(Variable("a"), Fraction(1, Variable("a")))
    expr2 = Number(1)

    test_print_message(expr1, expr2)
    result2 = apply_inverse_mul(expr1)
    print(result2)  # Output: 1
    assert result2 == expr2




def main():
    #test_AA()
    #test_AM()
    #test_CA()
    #test_CM()
    #test_D()
    test_ID()
    test_IM()
    #test_IvA()
    test_IvM()

main()