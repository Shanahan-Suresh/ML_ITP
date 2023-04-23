from inference_rules import *

#Modus Ponens Test
def test_mp():

    # Define premises and conclusion
    A = Variable("A")
    B = Variable("B")
    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = A
    conclusion = B

    # Apply modus ponens rule
    result = apply_modus_ponens(premise1, premise2)

    # Check if the rule application yielded the desired conclusion
    if result == conclusion:
        print("The modus ponens rule was applied successfully!")
    else:
        print("The modus ponens rule could not be applied.")


# Modus Tollens Test
def test_mt():

    A = Variable("A")
    B = Variable("B")
    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = UnaryOp(Connective.NOT, B)
    conclusion = UnaryOp(Connective.NOT, A)

    result = apply_modus_tollens(premise1, premise2)

    if result == conclusion:
        print("The modus tollens rule was applied successfully!")
    else:
        print("The modus tollens rule could not be applied.")

# Hypothetical Syllogism Test
def test_hs():
    A = Variable("A")
    B = Variable("B")
    C = Variable("C")

    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = BinaryOp(B, Connective.IMPLIES, C)
    conclusion = BinaryOp(A, Connective.IMPLIES, C)

    result = apply_hypothetical_syllogism(premise1, premise2)

    if result == conclusion:
        print("The hypothetical syllogism rule was applied successfully!")
    else:
        print("The hypothetical syllogism rule could not be applied.")

# Disjunctive Syllogism Test
def test_ds():
    a = Variable('a')
    b = Variable('b')

    expr1 = BinaryOp(a, Connective.OR, b)
    expr2 = UnaryOp(Connective.NOT, a)

    result = apply_disjunctive_syllogism(expr1, expr2)
    print(result)  # Output: b


# Additon Test
def test_a():
    A = Variable("A")
    B = Variable("B")

    # Test cases for apply_addition
    premise1 = A
    premise2 = BinaryOp(A, Connective.OR, B)
    result = apply_addition(premise1, premise2)
    print(result)
    assert result == premise2, "Test case 1 for apply_addition failed"

    premise1 = B
    premise2 = BinaryOp(A, Connective.OR, B)
    result = apply_addition(premise1, premise2)
    assert result == premise2, "Test case 2 for apply_addition failed"


# Simplification Test
def test_s():
    A = Variable("A")
    B = Variable("B")

    # Test cases for apply_simplification
    premise1 = BinaryOp(A, Connective.AND, B)
    premise2 = A
    result = apply_simplification(premise1, premise2)
    print(result)
    assert result == B, "Test case 1 for apply_simplification failed"

    premise1 = BinaryOp(A, Connective.AND, B)
    premise2 = B
    result = apply_simplification(premise1, premise2)
    assert result == A, "Test case 2 for apply_simplification failed"


# Resolution Test - NOT WORKING
def test_r():
    A = Variable("A")
    B = Variable("B")
    C = Variable("C")

    premise1 = Or(A, B)
    premise2 = Or(Not(B), C)
    conclusion = Or(A, C)

    print(apply_resolution(premise1, premise2, conclusion))  # Output: True


def main():
    #test_mp()
    #test_mt()
    #test_hs()
    #test_ds()
    #test_a()
    test_s()
    #test_r()


main()