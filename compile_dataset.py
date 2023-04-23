import json
import os

file_names = [
    "addition_proofs.json",
    "disjunctive_syllogism.json",
    "hypothetical_syllogism_proofs.json",
    "modus_ponens_proofs.json",
    "modus_tollens_proofs.json",
    "ponus_tollens_proofs.json",
    "simplification_proofs.json",
    "three_rule_proofs.json",
    "two_rule_proofs.json"
]

directory = "C:/Users/shana/source/repos/ML_ITP/Proofs Dataset/"

merged_proofs = []

for file_name in file_names:
    file_path = os.path.join(directory, file_name)
    with open(file_path, "r", encoding="utf-8") as infile:
        proofs = json.load(infile)
        merged_proofs.extend(proofs)

output_file_name = "compiled_proofs.json"
output_file_path = os.path.join(directory, output_file_name)

with open(output_file_path, "w", encoding="utf-8") as outfile:
    json.dump(merged_proofs, outfile, indent=2, ensure_ascii=False)

print(f"{len(merged_proofs)} merged proofs saved to {output_file_path}")

