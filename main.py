import copy
import re
from collections import defaultdict


token_specs = [
    ('RESERVEDWORD', r'#include|int|float|void|return|if|while|cin|cout|continue|break|using|namespace|std|main'),
    ('identifier', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('number', r'\d+'),
    ('string', r'"[^"]*"'),
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
    return 0 ;
}
"""

def check_wrong_initialization_and_assignment(code):
    errors = []
    # Regex to find variable declarations and assignments
    int_declaration_pattern = r'int\s+(\w+)\s*=\s*("[^"]*"|\d+)'
    string_declaration_pattern = r'string\s+(\w+)\s*=\s*(\d+)'
    int_assignment_pattern = r'(\w+)\s*=\s*("[^"]*")'
    string_assignment_pattern = r'(\w+)\s*=\s*(\d+)'

    # Check for wrong initializations
    int_declaration_matches = re.findall(int_declaration_pattern, code)
    for var_name, value in int_declaration_matches:
        if value.startswith('"'):
            errors.append(f"Wrong initialization: '{var_name}' is an int but is assigned a string value '{value}'")
    string_declaration_matches = re.findall(string_declaration_pattern, code)
    for var_name, value in string_declaration_matches:
        errors.append(f"Wrong initialization: '{var_name}' is a string but is assigned a numeric value '{value}'")

    # Check for wrong assignments
    int_assignment_matches = re.findall(int_assignment_pattern, code)
    for var_name, value in int_assignment_matches:
        # Check if the variable was declared as an int earlier
        if re.search(r'int\s+' + var_name + r'\s*[;=]', code):
            errors.append(f"Wrong assignment: '{var_name}' is an int but is assigned a string value '{value}'")

    string_assignment_matches = re.findall(string_assignment_pattern, code)
    for var_name, value in string_assignment_matches:
        # Check if the variable was declared as a string earlier
        if re.search(r'string\s+' + var_name + r'\s*[;=]', code):
            errors.append(f"SyntaxError: Wrong assignment: '{var_name}' is a string but is assigned a numeric value '{value}'")

        if errors:
            raise SyntaxError("\n".join(errors))

def check_missing_semicolon(code):
    lines = code.split('\n')
    for i, line in enumerate(lines):
        # Skip lines that are preprocessor directives, comments, or empty lines
        if line.strip().startswith('#') or line.strip().startswith('//') or line.strip() == '':
            continue
        # Check if the line ends with a semicolon
        if not line.strip().endswith(';') and not line.strip().endswith('{') and not line.strip().endswith('}'):
            raise SyntaxError(f"SyntaxError: Missing semicolon at line {i+1}: {line.strip()}")

try:
    check_wrong_initialization_and_assignment(cpp_code)
except SyntaxError as e:
    print(e)
    exit()
try:
    check_missing_semicolon(cpp_code)
except SyntaxError as e:
    print(e)
    exit()


# Tokenize the C++ code
tokens = tokenize(cpp_code)
print("tokens:", tokens)
not_sorted_tokens = tokens
# Define the order of token types
token_order = ['string', 'number', 'SYMBOL', 'identifier', 'RESERVEDWORD']

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
    'V': ['return number ;', 'ε'],
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
# parse_table['T'].pop('$')
        
# Nonrecursive Predictive Parser

number_list = []
string_list = []
identifier_list = []
neighbors_dict = {}

# Nonrecursive Predictive Parser
def nonrecursive_predictive_parser(tokens, parse_table, start_symbol):
    stack = ['$', start_symbol]  # Initialize stack with $ and start symbol
    

    # Prepare input tokens: use value for RESERVEDWORD and SYMBOL, otherwise use type
    input_tokens = []
    for token in tokens:
        token_type, token_value = token
        if token_type in ['RESERVEDWORD', 'SYMBOL']:
            input_tokens.append(token_value)  # Use value for RESERVEDWORD and SYMBOL
        else:
            if token_type == "number" :
                number_list.append(token_value)
            if token_type == "identifier" :
                identifier_list.append(token_value)
            if token_type == "string" :
                string_list.append(token_value)
            input_tokens.append(token_type)  # Use type for others (e.g., IDENTIFIER, NUMBER)
    input_tokens.append('$')  # Add end marker
    
    current_input = input_tokens[0]  # Current input token (value or type)
    
    output = []  # To store the sequence of productions
    index = 0  # Index to track the current input token
    # print("numList: ", number_list)
    # print("strList", string_list)
    # print("identifierList: ", identifier_list)

    
    print("Initial Stack:", stack)
    print("Input Tokens:", input_tokens)

    while stack:
        top = stack[-1]  # Get the top of the stack
        print(f"Stack: {stack}, Current Input: {current_input}")

        # If top of stack matches current input, pop and move to next input
        if top == current_input:
            if top == "identifier":
                output.append(f"{top} -> {identifier_list[0]}")
                neighbors_dict[top] = identifier_list[0]
                identifier_list.pop(0)
            elif top == "number":
                output.append(f"{top} -> {number_list[0]}")
                neighbors_dict[top] = number_list[0]
                number_list.pop(0)
            elif top == "string":
                output.append(f"{top} -> {string_list[0]}")
                string_list.pop(0)
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

#################b treeeeeeeeeeeeeeeeeee

