from formal_logic import *
from inference_rules import *
from proof_engine import dfs_search_inference_rules
import json
import random
import string

def random_premise(base_propositions, connectives, max_depth):
    if max_depth == 0 or random.random() < 0.2:
        return random.choice(base_propositions)
    
    connective = random.choice(connectives)
    left = random_premise(base_propositions, connectives, max_depth - 1)
    right = random_premise(base_propositions, connectives, max_depth - 1)
    
    return BinaryOp(left, connective, right)

def apply_rule_to_premises(rule_name, rule_func, premises):
    for premise1 in premises:
        for premise2 in premises:
            if premise1 != premise2:
                result = rule_func(premise1, premise2)
                if result:
                    return result
    return None

def generate_proof_with_three_rules(base_propositions, connectives, inference_rules, num_inferences=3, max_depth=3, premise_num=3):
    while True:
        # Generate a random set of premises and a conclusion
        premises = [random_premise(base_propositions, connectives, max_depth) for _ in range(1, premise_num)]
        current_exprs = set(premises)
        applied_rules = set()

        # Ensure the use of exactly three different inference rules
        attempts = 0
        max_attempts = 10 * num_inferences  # Adjust this value as needed
        while len(set(applied_rules)) < num_inferences and attempts < max_attempts:
            rule_name, rule_func = random.choice(inference_rules)
            result = apply_rule_to_premises(rule_name, rule_func, current_exprs)
            if result:
                current_exprs.add(result)
                applied_rules.add(rule_name)
            attempts += 1

        if len(set(applied_rules)) < num_inferences:
            continue

        # Choose a conclusion from the current expressions
        conclusion = random.choice(list(current_exprs))

        # Check if the conclusion can be reached using the premises and the inference rules
        _, resulting_exprs = dfs_search_inference_rules(set(premises), conclusion)
        if conclusion in resulting_exprs:
            return premises, conclusion, list(applied_rules)


def generate_proof_with_two_rules(base_propositions, connectives, inference_rules, num_inferences=2, max_depth=3, premise_num=3):
    while True:
        # Generate a random set of premises and a conclusion
        premises = [random_premise(base_propositions, connectives, max_depth) for _ in range(1, premise_num)]
        current_exprs = set(premises)
        applied_rules = set()

        # Ensure the use of at least two different inference rules
        attempts = 0
        max_attempts = 10 * num_inferences  # Adjust this value as needed
        while len(applied_rules) < num_inferences and attempts < max_attempts:
            rule_name, rule_func = random.choice(inference_rules)
            result = apply_rule_to_premises(rule_name, rule_func, current_exprs)
            if result:
                current_exprs.add(result)
                applied_rules.add(rule_name)
            attempts += 1

        if len(applied_rules) < num_inferences:
            continue

        # Choose a conclusion from the current expressions
        conclusion = random.choice(list(current_exprs))

        # Check if the conclusion can be reached using the premises and the inference rules
        _, resulting_exprs = dfs_search_inference_rules(set(premises), conclusion)
        if conclusion in resulting_exprs:
            return premises, conclusion, list(applied_rules)



def generate_proof(base_propositions, connectives, inference_rules, num_inferences=1, max_depth=3, premise_num=3):
    premises = [random_premise(base_propositions, connectives, max_depth) for _ in range(1, premise_num)]
    current_exprs = set(premises)
    applied_rules = []

    # Sample the rules without replacement
    selected_rules = random.sample(inference_rules, num_inferences)

    for rule_name, rule_func in selected_rules:
        result = apply_rule_to_premises(rule_name, rule_func, current_exprs)
        if result:
            current_exprs.add(result)
            applied_rules.append(rule_name)

    conclusion = random.choice(list(current_exprs))

    return premises, conclusion, applied_rules

def apply_generation():
    # Full list, alter as necessary
    #connectives = [Connective.AND, Connective.OR, Connective.IMPLIES, Connective.NOT, Connective.BICONDITIONAL, Connective.XOR]
    #inference_rules = [("modus_ponens", apply_modus_ponens), ("modus_tollens", apply_modus_tollens), ("hypothetical_syllogism", apply_hypothetical_syllogism), ("disjunctive_syllogism", apply_disjunctive_syllogism), ("addition", apply_addition), ("simplification", apply_simplification)]

    connectives = [Connective.AND, Connective.OR, Connective.IMPLIES, Connective.NOT, Connective.BICONDITIONAL, Connective.XOR]
    inference_rules = [("modus_ponens", apply_modus_ponens), ("modus_tollens", apply_modus_tollens), ("hypothetical_syllogism", apply_hypothetical_syllogism), ("addition", apply_addition), ("simplification", apply_simplification)]

    successful_proofs = []

    num_vars = 3 # Set the number of variables to use
    num_inferences = 3  # Change this to customize the number of inference rules used

    num_successful_proofs_required = 50  # Set the number of successful proofs you want to generate

    while len(successful_proofs) < num_successful_proofs_required:

        # Randomly select variable names
        selected_vars = random.sample(string.ascii_uppercase, num_vars)
    
        # Generate variable objects and base propositions
        variables = [Variable(ch) for ch in selected_vars]
        base_propositions = variables + [UnaryOp(Connective.NOT, v) for v in variables]

        # Proof generation
        premises, conclusion, applied_rules = generate_proof_with_three_rules(base_propositions, connectives, inference_rules, num_inferences=num_inferences)
        _, resulting_exprs = dfs_search_inference_rules(set(premises), conclusion)
    
        if conclusion in resulting_exprs and applied_rules:
            successful_proofs.append({
                "premises": [str(p) for p in premises],
                "conclusion": str(conclusion),
                "applied_rules": applied_rules
            })

    with open("three_rule_proofs.json", "w", encoding="utf-8") as outfile:
        json.dump(successful_proofs, outfile, indent=2, ensure_ascii=False)

    print(f"{len(successful_proofs)} successful proofs saved to successful_proofs.json")

def main():
     apply_generation()

main()


''' Single generation
premises, conclusion, applied_rules = generate_proof(base_propositions, connectives, inference_rules)
print("Premises:", premises)
print("Conclusion:", conclusion)
print("Applied Rules:", applied_rules)
'''