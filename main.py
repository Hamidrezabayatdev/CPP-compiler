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
        # current_node = node_stack[-1]
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
            # current_node.add_child(ParseTreeNode(input_tokens[index]))
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

for i in range(len(output)):
    output[i] = output[i].split(" -> ")
for i in range(len(output)):
    output[i][1] = output[i][1].split(" ")

# print("Parsing Output:")
# print(output)

#################b treeeeeeeeeeeeeeeeeee
class ParseTreeNode:
    def __init__(self, symbol, unique_id=None):
        self.symbol = symbol
        self.children = []
        self.unique_id = unique_id  # Unique identifier for handling repeated nodes

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"{self.symbol} (ID: {self.unique_id})"
    
    def show_children(self):
        return f"childrens of {self}: {self.children}"
    
    def show_root(self):
        return f"tree root is {self}"

    def last_occurrence(self, symbol):
        # Helper function for DFS traversal
        def dfs(node):
            nonlocal last_node
            if node.symbol == symbol:
                last_node = node  # Update last_node if the current node matches the symbol
            for child in node.children:
                dfs(child)  # Recursively traverse children

        last_node = None  # Initialize last_node to None
        dfs(self)  # Start DFS from the current node (root of the subtree)
        return last_node  # Return the last node with the matching symbol
# before build recommended fix
def build_parse_tree(output):
    # Initialize the root node with the start symbol
    root = ParseTreeNode("Start", unique_id=0)
    node_stack = ["Start"]
    current_id = 0  # Unique ID counter
    def find_last_occurrence_and_create_node(root, symbol):
        # Find the last occurrence of the symbol in the tree
        last_node = root.last_occurrence(symbol)

        if last_node:
            # Create a new node with the symbol and the unique_id of the last occurrence
            new_node = ParseTreeNode(symbol, unique_id=last_node.unique_id)
            return new_node.unique_id
        else:
            # If the symbol is not found, return None
            return current_id
        

    for index, production in enumerate(output):
        parent_symbol = production[0]
        children_symbols = production[1]

        # # Find the parent node in the stack
        # while node_stack and node_stack[-1].symbol in terminals:
        #     node_stack.pop()

        if not node_stack:
            raise ValueError(f"Invalid production sequence: No parent found for {parent_symbol}")
        current_node = node_stack[-1]
        while node_stack[-1] in terminals:
            node_stack.pop()
            if len(node_stack) == 0:
                return root
        # current_node = node_stack[-1]
        # while parent_symbol != current_node:
        #     node_stack.pop()
        current_node = node_stack[-1]
        if current_node == "Start":
            current_node = root
        else:
            current_node = ParseTreeNode(current_node, unique_id = find_last_occurrence_and_create_node(root, current_node))
        current_id += 1
        # Add children to the current node
        for symbol in children_symbols:
            if symbol == 'ε':
                continue  # Skip epsilon productions
            child_node = ParseTreeNode(symbol, unique_id=current_id)
            current_node.add_child(child_node)
            # node_stack.append(child_node)  # Push the child node onto the stack
            current_id += 1  # Increment unique ID

        # Pop the current node from the stack ONLY if it has no more children to process
        # This ensures that nodes like 'M' and 'N' are not popped prematurely

        node_stack.pop()
        print(current_node.show_children())
        for child in reversed(children_symbols):
            if child != "ε":
                node_stack.append(child)
        if len(node_stack) == 0:
            return root
    return root


from graphviz import Digraph

def export_parse_tree_to_png(root, filename="parse_tree"):
    dot = Digraph(comment="Parse Tree")

    def add_nodes_edges(node):
        # Add the current node to the graph
        dot.node(str(node.unique_id), label=node.symbol)
        # Recursively add children
        for child in node.children:
            dot.edge(str(node.unique_id), str(child.unique_id))
            add_nodes_edges(child)

    # Start adding nodes and edges from the root
    add_nodes_edges(root)

    # Render and save the graph as a PNG file
    dot.render(filename, format="png", cleanup=True)
    print(f"Parse tree exported to {filename}.png")


    # Build the parse tree from the output
parse_tree_root = build_parse_tree(output)

# Export the parse tree to a PNG file
export_parse_tree_to_png(parse_tree_root, filename="parse_tree")