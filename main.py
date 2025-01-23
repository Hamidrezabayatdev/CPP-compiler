import re
from collections import defaultdict


token_specs = [
    ('RESERVEDWORD', r'#include|int|float|void|return|if|while|cin|cout|continue|break|using|namespace|std|main'),
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
#include
using namespace std;
int main() {
    int x;
    int s = 0;
    while (x >= 0) {
        cin >> x;
        s = s + x;
    }
    cout << "sum=" << s;
    return 0;
}
"""
# Tokenize the C++ code
tokens = tokenize(cpp_code)
not_sorted_tokens = tokens 
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
    'N': ['using namespace std ;', 'ε'],
    'M': ['int main ( ) { T V }'],  # Updated to handle 'main', '(', and ')' as separate tokens
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
    'Loop': ['while ( Expression ) { T }'],
    'Input': ['cin >> identifier F ;'],
    'F': ['>> identifier F', 'ε'],
    'Output': ['cout << C H ;'],
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
                                if next_symbol not in follow[symbol]:
                                    follow[symbol].add(next_symbol)
                                    updated = True
                            else:
                                first_next = first_sets[next_symbol] - {'ε'}
                                if not first_next.issubset(follow[symbol]):
                                    follow[symbol].update(first_next)
                                    updated = True
                                if 'ε' in first_sets[next_symbol]:
                                    if not follow[non_terminal].issubset(follow[symbol]):
                                        follow[symbol].update(follow[non_terminal])
                                        updated = True
                        else:
                            if not follow[non_terminal].issubset(follow[symbol]):
                                follow[symbol].update(follow[non_terminal])
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
        
# Nonrecursive Predictive Parser
def nonrecursive_predictive_parser(tokens, parse_table, start_symbol):
    stack = ['$', start_symbol]  # Initialize stack with $ and start symbol
    input_tokens = [token[1] for token in tokens] + ['$']  # Extract token values and add end marker
    current_input = input_tokens[0]  # Current input token
    output = []  # To store the sequence of productions
    index = 0  # Index to track the current input token
    
    while stack:
        top = stack[-1]  # Get the top of the stack

        # If top of stack matches current input, pop and move to next input
        if top == current_input:
            stack.pop()  # Pop the matched symbol
            index += 1  # Move to the next input token
            current_input = input_tokens[index] if index < len(input_tokens) else '$'
        
        # If top is a terminal but doesn't match input, raise an error
        elif top in terminals:
            raise SyntaxError(f"Unexpected terminal {top}. Expected {current_input}")
        
        # If top is a non-terminal, use the parse table to find the production
        elif top in non_terminals:
            if current_input in parse_table[top]:
                production = parse_table[top][current_input]  # Get the production
                stack.pop()  # Pop the non-terminal
                if production != 'ε':  # If production is not epsilon, push symbols onto the stack
                    for symbol in reversed(production.split()):
                        stack.append(symbol)
                output.append(f"{top} -> {production}")  # Add production to output
            else:
                raise SyntaxError(f"No production found for {top} with input {current_input}")
        
        # If top is neither a terminal nor a non-terminal, raise an error
        else:
            raise SyntaxError(f"Invalid symbol {top} on stack")

    return output

# Parse the tokenized input
try:
    output = nonrecursive_predictive_parser(tokens, parse_table, 'Start')
    print("Parsing Output:")
    for line in output:
        print(line)
except SyntaxError as e:
    print(f"Syntax Error: {e}")
    
