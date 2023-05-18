import tkinter as tk
from tkinter import messagebox, font, Toplevel
from parser import parse_expression
from proof_engine import dfs_search_inference_rules
from tokenizer import custom_tokenizer
from load_models import *

def submit_expression():
    user_input = input_entry.get()
    if user_input.lower() == "quit":
        expression_checker.destroy()
        return
    try:
        expr = parse_expression(user_input)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Parsed expression: {expr}")
    except Exception as e:
        messagebox.showerror("Error", f"Error parsing expression: {e}")

def run_proof_search():
    premises_input = premises_entry.get()
    conclusion_input = conclusion_entry.get()

    if not premises_input or not conclusion_input:
        messagebox.showerror("Error", "Please enter premises and a conclusion.")
        return

    premises_list = premises_input.split(',')
    premises = set()
    for premise_str in premises_list:
        premise_str = premise_str.strip()
        premise = parse_expression(premise_str)
        premises.add(premise)

    conclusion = parse_expression(conclusion_input)

    applied_rules, result = dfs_search_inference_rules(premises, conclusion)

    if applied_rules:
        message = f"Conclusion can be reached using the following rules: {applied_rules}\nConclusion reached: {result[0]}"
    else:
        message = "No valid sequence of inference rules found."

    messagebox.showinfo("Proof Search Result", message)

def display_predictions():
    premises_input = premises_entry.get().split(",")
    conclusion_input = conclusion_entry.get()
    knn_predicted_rules = knn_predict_inference_rules(premises_input, conclusion_input)
    dt_predicted_rules = decision_tree_predict_inference_rules(premises_input, conclusion_input)
    rf_predicted_rules = random_forest_predict_inference_rules(premises_input, conclusion_input)
    #ann_predicted_rules = ann_predict_inference_rules(premises_input, conclusion_input)
    cnn_predicted_rules = cnn_predict_inference_rules(premises_input, conclusion_input)

    if not rf_predicted_rules:
        rf_predicted_rules_str = "No rule predicted"

    else :
        # Convert the tuple of predicted rules to a string
        rf_predicted_rules_str = ', '.join(rf_predicted_rules)

    knn_predicted_rules_str = ', '.join(knn_predicted_rules)
    dt_predicted_rules_str = ', '.join(dt_predicted_rules)
    #ann_predicted_rules_str = ', '.join(ann_predicted_rules)
    cnn_predicted_rules_str = ', '.join(cnn_predicted_rules)
    
    print('Im here', rf_predicted_rules, 'yes')
    # Create a new top-level window
    prediction_window = Toplevel(root)
    prediction_window.title("Suggest Proof Steps")
    
    # Add labels to display the predictions
    knn_prediction_label = tk.Label(prediction_window, text=f"KNN Prediction: {knn_predicted_rules_str}", bg="#1e1e2f", fg=font_color)
    knn_prediction_label.pack(padx=20, pady=10)

    dt_prediction_label = tk.Label(prediction_window, text=f"Decision Tree Prediction: {dt_predicted_rules_str}", bg="#1e1e2f", fg=font_color)
    dt_prediction_label.pack(padx=20, pady=10)

    rf_prediction_label = tk.Label(prediction_window, text=f"Random Forest Prediction: {rf_predicted_rules_str}", bg="#1e1e2f", fg=font_color)
    rf_prediction_label.pack(padx=20, pady=10)

    #ann_prediction_label = tk.Label(prediction_window, text=f"Feedforward Network Prediction: {ann_predicted_rules_str}", bg="#1e1e2f", fg=font_color)
    #ann_prediction_label.pack(padx=20, pady=10)

    cnn_prediction_label = tk.Label(prediction_window, text=f"Convolutional Network Prediction: {cnn_predicted_rules_str}", bg="#1e1e2f", fg=font_color)
    cnn_prediction_label.pack(padx=20, pady=10)




def open_expression_checker():
    global expression_checker
    global input_entry
    global result_text

    def copy_result():
        result_text.clipboard_clear()
        result_text.clipboard_append(result_text.get(1.0, tk.END))

        copy_button = tk.Button(expression_checker, text="Copy", command=copy_result, bg=button_bg, fg=button_fg)
        copy_button.pack(pady=10)

    
    expression_checker = tk.Toplevel(root)
    expression_checker.title("Expression Checker")
    expression_checker.geometry(f"{int(screen_width * 0.5)}x{int(screen_height * 0.5)}+{int(screen_width * 0.1)}+{int(screen_height * 0.1)}")
    expression_checker.configure(bg="#1e1e2f")


    instructions_label = tk.Label(expression_checker, text="Enter an expression using the following syntax:", bg="#1e1e2f", fg=font_color, font=app_font, wraplength=int(screen_width * 0.8) - 40)
    instructions_label.pack(pady=10)

    syntax_label = tk.Label(expression_checker, text="Variables: a, b, c, ...\nConnectives: ∧ (and), ∨ (or), ¬ (not), → (implies), ↔ (biconditional), ⊕ (xor), +, -, *, /, =, - (negation)\nParentheses: (, )", bg="#1e1e2f", fg=font_color, font=app_font, wraplength=int(screen_width * 0.8) - 40)
    syntax_label.pack(pady=10)

    input_label = tk.Label(expression_checker, text="Enter an expression:", bg="#1e1e2f", fg=font_color, font=app_font)
    input_label.pack(pady=10)

    input_entry = tk.Entry(expression_checker, width=50, bg="#1e1e2f", fg=font_color, insertbackground=font_color, font=app_font)
    input_entry.pack(pady=10)

    submit_button = tk.Button(expression_checker, text="Submit", command=submit_expression, bg=button_bg, fg=button_fg, font=app_font)
    submit_button.pack(pady=10)

    result_text = tk.Text(expression_checker, width=40, height=3, bg="#1e1e2f", fg=font_color, wrap="word", font=app_font)
    result_text.pack(pady=10)

    quit_button = tk.Button(expression_checker, text="Quit", command=expression_checker.destroy, bg=button_bg, fg=button_fg, font=app_font)
    quit_button.pack(pady=10)


root = tk.Tk()
root.title("Interactive Theorem Prover")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{int(screen_width * 0.6)}x{int(screen_height * 0.5)}+{int(screen_width * 0.1)}+{int(screen_height * 0.1)}")
root.configure(bg="#1e1e2f")

font_color = "#ffffff"
button_bg = "#2e2e3e"
button_fg = "#ffffff"
app_font = font.nametofont("TkDefaultFont").copy()
app_font.config(size=14, family="Helvetica")

premises_label = tk.Label(root, text="Enter premises (separated by commas):", bg="#1e1e2f", fg=font_color, font=app_font)
premises_label.pack(pady=10)

premises_entry = tk.Entry(root, width=50, bg="#1e1e2f", fg=font_color, insertbackground=font_color, font=app_font)
premises_entry.pack(pady=10)

conclusions_label = tk.Label(root, text="Enter conclusion:", bg="#1e1e2f", fg=font_color, font=app_font)
conclusions_label.pack(pady=10)

conclusion_entry = tk.Entry(root, width=50, bg="#1e1e2f", fg=font_color, insertbackground=font_color, font=app_font)
conclusion_entry.pack(pady=10)

proof_search_button = tk.Button(root, text="DFS Proof Search", command=run_proof_search, bg=button_bg, fg=button_fg, font=app_font)
proof_search_button.pack(pady=10)

prediction_button = tk.Button(root, text="Suggest Proof Steps", command=display_predictions, bg=button_bg, fg=button_fg, font=app_font)
prediction_button.pack(pady=10)

expr_checker_button = tk.Button(root, text="Expression Checker", command=open_expression_checker, bg=button_bg, fg=button_fg, font=app_font)
expr_checker_button.pack(pady=10)

quit_button = tk.Button(root, text="Quit", command=root.destroy, bg=button_bg, fg=button_fg, font=app_font)
quit_button.pack(pady=10)

root.mainloop()
