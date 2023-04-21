from inference_rules import *
import itertools

# List of all inference rules
all_inference_rules = [
    ('modus_ponens', apply_modus_ponens),
    ('modus_tollens', apply_modus_tollens),
    ('hypothetical_syllogism', apply_hypothetical_syllogism),
    ('disjunctive_syllogism', apply_disjunctive_syllogism),
    ('addition', apply_addition),
    ('simplification', apply_simplification),
    #('resolution', apply_resolution),
    ('associativity_add', apply_associativity_add),
    ('associativity_mul', apply_associativity_mul),
    ('commutativity_add', apply_commutativity_add),
    ('commutativity_mul', apply_commutativity_mul),
    ('distributivity', apply_distributivity),
    ('identity_add', apply_identity_add),
    ('identity_mul', apply_identity_mul),
    ('inverse_add', apply_inverse_add),
    ('inverse_mul', apply_inverse_mul),
]

# Function to carry out specific inference rules
def apply_rule(premise1, premise2, rule):
    rule_name, rule_func = rule
    return rule_func(premise1, premise2)

def search_inference_rules(premise1, premise2):
    for rule_name, rule_func in all_inference_rules:
        result = apply_rule(premise1, premise2, (rule_name, rule_func))
        if result:
            return rule_name, result
    return None, None

def dfs_search_inference_rules(premises, conclusion, depth=0, max_depth=15, visited=None, applied_rules=None):
    if visited is None:
        visited = set()
        applied_rules = []

    premises_tuple = tuple(premises)
    if premises_tuple in visited:
        return None, []

    visited.add(premises_tuple)

    #print(f"Depth: {depth}, Premises: {premises}, Applied Rules: {applied_rules}")  # Print 1

    for rule_name, rule_func in all_inference_rules:
        if getattr(rule_func, 'single_premise', False):
            for premise in premises:
                result = rule_func(premise)
                if result:
                    #print(f"Depth: {depth}, Premises: {premises}, Result: {result} Applied Rules: {applied_rules}")  # Print 2
                    #print(f"Indended conclusion: {conclusion}")
                    if result == conclusion:
                        return applied_rules + [rule_name], [result]

                    if depth < max_depth:
                        next_applied_rules, next_result = dfs_search_inference_rules(
                            premises.union({result}), conclusion, depth + 1, max_depth, visited, applied_rules + [rule_name]
                        )
                        if next_result:
                            return next_applied_rules, [result] + next_result
        else:
            for premise1 in premises:
                for premise2 in premises:
                    if premise1 == premise2:
                        continue

                    result = apply_rule(premise1, premise2, (rule_name, rule_func))

                    if result:
                        if result == conclusion:
                            return applied_rules + [rule_name], [result]

                        if depth < max_depth:
                            next_applied_rules, next_result = dfs_search_inference_rules(
                                premises.union({result}), conclusion, depth + 1, max_depth, visited, applied_rules + [rule_name]
                            )
                            if next_result:
                                return next_applied_rules, [result] + next_result

    return None, []






