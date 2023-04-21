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
    expr1 = BinaryOp(BinaryOp(A, Connective.ADD, B), Connective.ADD, BinaryOp(C, Connective.ADD, D))
    expr2 = BinaryOp(BinaryOp(C, Connective.ADD, D), Connective.ADD, E)

    # Conclusion
    conclusion = BinaryOp(A, Connective.ADD, BinaryOp(B, Connective.ADD, BinaryOp(C, Connective.ADD, BinaryOp(D, Connective.ADD, E))))

    # Run the DFS search
    #applied_rules, resulting_exprs = dfs_search_inference_rules({expr1, expr2}, conclusion)

    # Check the result
    #print("Applied rules:", applied_rules)
    #print("Resulting expressions:", resulting_exprs)


def main():
    #search_test()
    dfs_search_test()

main()
