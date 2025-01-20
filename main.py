import re
from collections import defaultdict


token_specs = [
    ('RESERVEDWORD', r'#include\s*<[^>]+>|int|float|void|return|if|while|cin|cout|continue|break|using|namespace|std|main'),
    ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('NUMBER', r'\d+'),
    ('STRING', r'"[^"]*"'),
    ('SYMBOL', r'<<|>>|>=|<=|==|!=|\+|\-|\*|\/|\(|\)|\{|\}|=|,|;|>|<|\|\||&&|!'),
    ('WHITESPACE', r'\s+'),  # Ignore whitespace
    ('UNKNOWN', r'.'),  # Catch-all for unknown characters
]

# Combine all regular expressions into one
token_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specs)

# Tokenize function
def tokenize(code):
    tokens = []
    for match in re.finditer(token_regex, code):
        kind = match.lastgroup
        value = match.group()
        if kind == 'WHITESPACE':
            continue  # Skip whitespace
        elif kind == 'UNKNOWN':
            raise SyntaxError(f'Unknown token: {value}')
        tokens.append((kind, value))
    return tokens

# Example C++ code
cpp_code = """
#include <iostream>
using namespace std;
int main() {
    int x;
    int s = 0, t = 10;
    while (t >= 0) {
        cin >> x;
        t = t - 1;
        s = s + x;
    }
    cout << "sum=" << s;
    return 0;
}
"""

# Tokenize the C++ code
tokens = tokenize(cpp_code)

# Define the order of token types
token_order = ['STRING', 'NUMBER', 'SYMBOL', 'IDENTIFIER', 'RESERVEDWORD']

# Function to build the Token Table
def build_token_table(tokens):
    token_table = []
    for token_type, token_value in tokens:
        token_table.append({'Token Name': token_type, 'Token Value': token_value})
    return token_table

# Function to sort the Token Table based on token order and ASCII value
def sort_token_table(token_table, token_order):
    # Create a dictionary to map token types to their priority
    priority = {token_type: idx for idx, token_type in enumerate(token_order)}
    
    # Sort the token table:
    # 1. First by token priority (to group tokens of the same type together)
    # 2. Then by Token Value in ASCII order (to sort tokens within the same group)
    sorted_table = sorted(token_table, key=lambda x: (priority[x['Token Name']], x['Token Value']))
    return sorted_table

# Build the Token Table
token_table = build_token_table(tokens)

# Sort the Token Table
sorted_token_table = sort_token_table(token_table, token_order)

# Print the Token Table
print("Token Table:")
print("{:<15} {:<15}".format("Token Name", "Token Value"))
print("-" * 30)
for entry in sorted_token_table:
    print("{:<15} {:<15}".format(entry['Token Name'], entry['Token Value']))
    
# Define the CFG rules
cfg_rules = {
    'Start': ['S N M'],
    'S': ['#include S', 'ε'],
    'N': ['using namespace std;', 'ε'],
    'M': ['int main() { T V }'],
    'T': ['Id T', 'L T', 'Loop T', 'Input T', 'Output T', 'ε'],
    'V': ['return 0;', 'ε'],
    'Id': ['int L', 'float L'],
    'L': ['identifier Assign Z'],
    'Assign': ['= Operation', 'ε'],
    'Z': [', identifier Assign Z', ';'],
    'Operation': ['number P', 'identifier P'],
    'P': ['O W P', 'ε'],
    'O': ['+', '-', '*'],
    'W': ['number', 'identifier'],
    'Expression': ['Operation K Operation'],
    'K': ['==', '>=', '<=', '!='],
    'Loop': ['while (Expression) { T }'],
    'Input': ['cin >> identifier F;'],
    'F': ['>> identifier F', 'ε'],
    'Output': ['cout << C H;'],
    'H': ['<< C H', 'ε'],
    'C': ['number', 'string', 'identifier'],
}


terminals = {
    '#include', 'using', 'namespace', 'std', 'int', 'float', 'void', 'return', 'if', 'while', 'cin', 'cout', 
    'continue', 'break', 'main', 'identifier', 'number', 'string', '<<', '>>', '>=', '<=', '==', '!=', 
    '+', '-', '*', '/', '(', ')', '{', '}', '=', ',', ';', '>', '<', '||', '&&', '!', '"'
}

non_terminals = set(cfg_rules.keys())

# Function to compute FIRST sets
def compute_first(cfg_rules, terminals, non_terminals):
    first = defaultdict(set)
    
    def first_of(symbol):
        if symbol in terminals:
            return {symbol}
        if symbol in first:
            return first[symbol]
        
        first[symbol] = set()
        for production in cfg_rules[symbol]:
            # print(production)
            for i, sym in enumerate(production.split()):
                if sym == 'ε':
                    first[symbol].add('ε')
                    break
                first_sym = first_of(sym)
                first[symbol].update(first_sym - {'ε'})
                if 'ε' not in first_sym:
                    break
                if i == len(production.split()) - 1:
                    first[symbol].add('ε')
        return first[symbol]
    
    for non_terminal in non_terminals:
        first_of(non_terminal)
    
    return first

# Compute FIRST sets
first_sets = compute_first(cfg_rules, terminals, non_terminals)

def compute_follow(cfg_rules, first_sets, start_symbol):
    follow = defaultdict(set)
    follow[start_symbol].add('$')  # Rule 1: Add $ to FOLLOW of start symbol
    
    while True:
        updated = False
        
        for non_terminal in cfg_rules:
            for production in cfg_rules[non_terminal]:
                production_symbols = production.split()
                
                for i, symbol in enumerate(production_symbols):
                    if symbol in non_terminals:
                        # Rule 2: A -> αBβ
                        if i + 1 < len(production_symbols):
                            next_symbol = production_symbols[i + 1]
                            if next_symbol in terminals:
                                follow[symbol].add(next_symbol)
                            else:
                                follow[symbol].update(first_sets[next_symbol] - {'ε'})
                                if 'ε' in first_sets[next_symbol]:
                                    follow[symbol].update(follow[non_terminal])
                        # Rule 3: A -> αB or A -> αBβ where β derives ε
                        else:
                            follow[symbol].update(follow[non_terminal])
                        
                        if follow[symbol] != follow[symbol]:
                            updated = True
        
        if not updated:
            break
    
    return follow

# Compute FOLLOW sets
follow_sets = compute_follow(cfg_rules, first_sets, 'Start')

# Print FIRST sets
print("FIRST Sets:")
for non_terminal, first_set in first_sets.items():
    print(f'FIRST({non_terminal}) = {first_set}')

# Print FOLLOW sets
print("FOLLOW Sets:")
for non_terminal, follow_set in follow_sets.items():
    print(f'FOLLOW({non_terminal}) = {follow_set}')


# Function to build the Parse Table
def build_parse_table(cfg_rules, first_sets, follow_sets, terminals, non_terminals):
    parse_table = defaultdict(dict)
    
    for non_terminal in non_terminals:
        for production in cfg_rules[non_terminal]:
            first_alpha = set()
            production_symbols = production.split()
            
            # Compute FIRST(α) for the production
            for symbol in production_symbols:
                if symbol in terminals:
                    first_alpha.add(symbol)
                    break
                elif symbol in non_terminals:
                    first_alpha.update(first_sets[symbol] - {'ε'})
                    if 'ε' not in first_sets[symbol]:
                        break
                elif symbol == 'ε':
                    first_alpha.add('ε')
                    break
            else:
                first_alpha.add('ε')
            
            # Add production to M[A, a] for each terminal in FIRST(α)
            for terminal in first_alpha:
                if terminal != 'ε':
                    parse_table[non_terminal][terminal] = production
            
            # If ε is in FIRST(α), add production to M[A, b] for each terminal in FOLLOW(A)
            if 'ε' in first_alpha:
                for terminal in follow_sets[non_terminal]:
                    parse_table[non_terminal][terminal] = production
    
    return parse_table

# Build the Parse Table
parse_table = build_parse_table(cfg_rules, first_sets, follow_sets, terminals, non_terminals)

# Print the Parse Table
print("Parse Table:")
for non_terminal in non_terminals:
    print(f"{non_terminal}:")
    for terminal, production in parse_table[non_terminal].items():
        print(f"  {terminal} -> {production}")
        
# def nonrecursive_predictive_parser(input_string, parse_table, start_symbol):
#     stack = ['$', start_symbol]  # Initialize stack with $ and start symbol
#     input_pointer = 0  # Initialize input pointer
#     input_tokens = input_string.split()  # Tokenize input string
    
#     print("Parsing Steps:")
#     while stack:
#         print(f"Stack: {stack}, Input: {' '.join(input_tokens[input_pointer:])}")
#         top = stack[-1]  # Top of the stack
#         current_input = input_tokens[input_pointer] if input_pointer < len(input_tokens) else '$'
        
#         if top in terminals:
#             if top == current_input:
#                 stack.pop()  # Match terminal
#                 input_pointer += 1  # Move to next input symbol
#             else:
#                 print(f"Error: Expected {top}, found {current_input}")
#                 return False
#         elif top in non_terminals:
#             if current_input in parse_table[top]:
#                 production = parse_table[top][current_input]
#                 stack.pop()  # Pop non-terminal
#                 if production != 'ε':
#                     # Push production symbols onto the stack in reverse order
#                     for symbol in reversed(production.split()):
#                         stack.append(symbol)
#             else:
#                 print(f"Error: No production for {top} with input {current_input}")
#                 return False
#         else:
#             print(f"Error: Invalid symbol {top} on stack")
#             return False
    
#     if input_pointer == len(input_tokens):
#         print("Input string is valid!")
#         return True
#     else:
#         print("Error: Input string is not fully parsed")
#         return False


# # Example Input
# input_string = "#include <iostream> using namespace std; int main() { int x; int s = 0; while (x >= 0) { cin >> x; s = s + x; } cout << s; return 0; }"

# # Run the parser
# nonrecursive_predictive_parser(input_string, parse_table, 'Start')