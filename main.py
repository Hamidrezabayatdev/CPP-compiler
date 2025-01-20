import re

token_specs = [
    ('RESERVEDWORD', r'int|float|void|return|if|while|cin|cout|continue|break|using|namespace|std|main|#include'),
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