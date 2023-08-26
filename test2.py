from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

# Python代码
python_code = """
def greet(name):
    print("Hello, " + name)
    #hsksd
greet("Alice")
"""

# 使用PythonLexer进行分词
lexer = PythonLexer()
tokens = list(lexer.get_tokens(python_code))

# 打印分词结果
for token_type, token_value in tokens:
    print(f"{token_type}: {token_value}")
