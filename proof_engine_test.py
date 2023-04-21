from proof_engine import *

def search_test():
    A = Variable("A")
    B = Variable("B")
    C = Variable("C")

    # Test Modus Ponens
    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = A
    rule_name, result = dfs_search_inference_rules(premise1, premise2)
    print(f"Applied rule: {rule_name}, Result: {result}")

    # Test Modus Tollens
    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = UnaryOp(Connective.NOT, B)
    rule_name, result = search_inference_rules(premise1, premise2)
    print(f"Applied rule: {rule_name}, Result: {result}")

    # Test HS
    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = BinaryOp(B, Connective.IMPLIES, C)
    rule_name, result = search_inference_rules(premise1, premise2)
    print(f"Applied rule: {rule_name}, Result: {result}")

    # Test Addition
    premise1 = A
    premise2 = BinaryOp(A, Connective.OR, B)
    rule_name, result = search_inference_rules(premise1, premise2)
    print(f"Applied rule: {rule_name}, Result: {result}")

    # Test DS
    premise1 = BinaryOp(A, Connective.OR, B)
    premise2 = UnaryOp(Connective.NOT, A)
    rule_name, result = search_inference_rules(premise1, premise2)
    print(f"Applied rule: {rule_name}, Result: {result}")

    # Test Simplification
    premise1 = BinaryOp(A, Connective.AND, B)
    premise2 = A
    rule_name, result = search_inference_rules(premise1, premise2)
    print(f"Applied rule: {rule_name}, Result: {result}")

def dfs_search_test():
    A = Variable('A')
    B = Variable('B')
    C = Variable('C')
    D = Variable('D')
    E = Variable('E')


    # Modus Pollus
    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = A
    conclusion = B

    # Call the dfs_search_inference_rules function
    result = dfs_search_inference_rules({premise1, premise2}, conclusion)

    # Print the result
    print("Result:", result)

    # MP and DS
    premise1 = BinaryOp(A, Connective.IMPLIES, B)
    premise2 = A
    premise3 = BinaryOp(B, Connective.OR, C)
    premise4 = UnaryOp(Connective.NOT, B)

    conclusion = C

    # Call the dfs_search_inference_rules function
    result = dfs_search_inference_rules({premise1, premise2, premise3, premise4}, conclusion)

    # Print the result
    print("Result:", result)

    expr1 = BinaryOp(BinaryOp(A, Connective.ADD, B), Connective.ADD, C)

    # Conclusion
    conclusion = BinaryOp(A, Connective.ADD, BinaryOp(B, Connective.ADD, C))

    premises = {expr1}
    applied_rules, result = dfs_search_inference_rules(premises, conclusion)

    print("Applied Rules:", applied_rules)
    print("Result:", result)

    # 2 Associativity Add
    expr1 = BinaryOp(BinaryOp(BinaryOp(A, Connective.ADD, B), Connective.ADD, C), Connective.ADD, D)

    # Conclusion
    conclusion = BinaryOp(A, Connective.ADD, BinaryOp(B, Connective.ADD, BinaryOp(C, Connective.ADD, D)))

    # Run the DFS search
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)

    # Check the result
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    # Associativity Mul
    expr1 = BinaryOp(BinaryOp(A, Connective.MUL, B), Connective.MUL, BinaryOp(C, Connective.MUL, D))
    conclusion = BinaryOp(A, Connective.MUL, BinaryOp(B, Connective.MUL, BinaryOp(C, Connective.MUL, D)))
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    # Commutative Add
    expr1 = BinaryOp(A, Connective.ADD, B)
    conclusion = BinaryOp(B, Connective.ADD, A)
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    # Commutative Mul
    expr1 = BinaryOp(A, Connective.MUL, B)
    conclusion = BinaryOp(B, Connective.MUL, A)
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    # Distributivity
    expr1 = BinaryOp(A, Connective.MUL, BinaryOp(B, Connective.ADD, C))
    conclusion = BinaryOp(BinaryOp(A, Connective.MUL, B), Connective.ADD, BinaryOp(A, Connective.MUL, C))
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    # Identity Add
    X = Variable("X")
    expr1 = BinaryOp(Number(0), Connective.ADD, X)
    conclusion = X
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    # Identity Mul
    Y = Variable("Y")
    expr1 = BinaryOp(Number(1), Connective.MUL, Y)
    conclusion = Y
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    # Inverse Add
    Z = Variable("Z")
    expr1 = BinaryOp(UnaryOp(Connective.NEG, Z), Connective.ADD, Z)
    conclusion = Number(0)
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)


   # Inverse Add
    W = Variable("W")
    expr1 = BinaryOp(Fraction(1, W), Connective.MUL, W)
    conclusion = Number(1)
    applied_rules, resulting_exprs = dfs_search_inference_rules({expr1}, conclusion)
    print("Applied rules:", applied_rules)
    print("Resulting expressions:", resulting_exprs)

    



def main():
    #search_test()
    dfs_search_test()

main()
