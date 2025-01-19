import re

# Define regular expressions for tokens
token_specs = [
    ('RESERVEDWORD', r'int|float|void|return|if|while|cin|cout|continue|break|include|using|namespace|std|main'),
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

# Print the tokens
for token in tokens:
    print(token)