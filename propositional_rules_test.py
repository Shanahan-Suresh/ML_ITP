from inference_rules import *

#Modus Ponens Test
def test_mp():

    A = Variable("A")
    B = Variable("B")

    premises = [A, Implies(A, B)]
    conclusion = B

    print(apply_modus_ponens(premises, conclusion))  # Output: True

# Modus Tollens Test
def test_mt():

    A = Variable("A")
    B = Variable("B")

    premises = [UnaryOp(Connective.NOT, B), Implies(A, B)]
    conclusion = UnaryOp(Connective.NOT, A)

    print(apply_modus_tollens(premises, conclusion))  # Output: True

# Hypothetical Syllogism Test
def test_hs():
    A = Variable("A")
    B = Variable("B")
    C = Variable("C")

    premises = [Implies(A, B), Implies(B, C)]
    conclusion = Implies(A, C)

    print(apply_hypothetical_syllogism(premises, conclusion))  # Output: True

# Additon Test
def test_a():
    A = Variable("A")
    B = Variable("B")

    premise = A
    conclusion = Or(A, B)

    print(apply_addition(premise, conclusion))  # Output: True


# Simplification Test
def test_s():
    A = Variable("A")
    B = Variable("B")

    premise = And(A, B)
    conclusion = A

    print(apply_simplification(premise, conclusion))  # Output: True


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
    #test_a()
    #test_s()
    test_r()


main()