import ast
import operator as op

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_eval_expr(node):
    """Safely evaluates mathematical AST nodes without raw eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left)
        right = safe_eval_expr(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            if op_type == ast.Div and right == 0:
                raise ValueError("Division by zero is undefined.")
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval_expr(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError("Invalid mathematical expression.")


def calculate_math(expression: str) -> str:
    """Safely evaluates standard arithmetic expressions using AST parsing."""
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = safe_eval_expr(parsed.body)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_current_weather(location: str) -> str:
    """Returns simulated weather data for a location."""
    if not location or not isinstance(location, str):
        return "Error: Invalid location provided."

    mock_db = {
        "baku": {"temp_c": 28, "condition": "Sunny", "humidity": 55},
        "sumgayit": {"temp_c": 26, "condition": "Clear", "humidity": 60},
        "london": {"temp_c": 18, "condition": "Rainy", "humidity": 80},
    }
    data = mock_db.get(
        location.strip().lower(),
        {"temp_c": 22, "condition": "Partly Cloudy", "humidity": 50},
    )

    return (
        f"[SIMULATED DATA] Weather in {location.title()}: "
        f"{data['temp_c']}°C, Condition: {data['condition']}, Humidity: {data['humidity']}%"
    )


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Converts temperature between Celsius ('C') and Fahrenheit ('F')."""
    from_u, to_u = from_unit.upper(), to_unit.upper()
    if from_u not in ["C", "F"] or to_u not in ["C", "F"]:
        return f"Error: Unsupported unit conversion from '{from_unit}' to '{to_unit}'."

    if from_u == to_u:
        return f"{value:.2f} °{to_u}"

    res = (value * 9 / 5) + 32 if from_u == "C" else (value - 32) * 5 / 9
    return f"{value} °{from_u} = {res:.2f} °{to_u}"


TOOL_MAP = {
    "get_current_weather": get_current_weather,
    "convert_temperature": convert_temperature,
    "calculate_math": calculate_math,
}

GROQ_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Fetch simulated current weather details for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "Convert temperature between Celsius ('C') and Fahrenheit ('F').",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Temperature value",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "'C' or 'F'",
                    },
                    "to_unit": {"type": "string", "description": "'C' or 'F'"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Perform basic arithmetic calculations safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic string (e.g., '28 * 9 / 5 + 32')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]