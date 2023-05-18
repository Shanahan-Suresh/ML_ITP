from parser import parse_expression
from proof_engine import dfs_search_inference_rules

def print_instructions():
    print("Interactive Theorem Prover Command-Line Interface")
    print("Enter expressions using the following syntax:")
    print("  Variables: a, b, c, ...")
    print("  Connectives: ∧ (and), ∨ (or), ¬ (not), → (implies), ↔ (biconditional), ⊕ (xor), +, -, *, /, =, - (negation)")
    print("  Parentheses: (, )")
    print("Type 'quit' to exit.\n")

def get_parsed_expression(prompt):
    while True:
        user_input = input(prompt)
        if user_input.lower() == "quit":
            return None
        try:
            expr = parse_expression(user_input)
            return expr
        except Exception as e:
            print(f"Error parsing expression: {e}")

def get_number_of_premises():
    while True:
        num_premises = input("Enter the number of premises: ")
        if num_premises.isdigit():
            return int(num_premises)
        else:
            print("Invalid input. Please enter a positive integer.")

if __name__ == "__main__":
    print_instructions()
    while True:
        num_premises = get_number_of_premises()
        premises = set()
        for i in range(num_premises):
            premise = get_parsed_expression(f"Enter premise {i+1} or type 'quit' to exit: ")
            if premise is None:
                break
            premises.add(premise)

        conclusion = get_parsed_expression("Enter the conclusion or type 'quit' to exit: ")
        if conclusion is None:
            break

        applied_rules, result = dfs_search_inference_rules(premises, conclusion)

        if applied_rules:
            print(f"Conclusion can be reached using the following rules: {applied_rules}")
            print(f"Conclusion reached: {result[0]}")
        else:
            print("No valid sequence of inference rules found.")
