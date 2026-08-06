import math
import json


def calculator(expression: str) -> str:
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def get_weather(location: str) -> str:
    loc_lower = location.lower().strip()
    if "baku" in loc_lower:
        return "The current weather in Baku is Sunny with a temperature of 29.4°C."
    elif "tokyo" in loc_lower:
        return "The current weather in Tokyo is Rainy with a temperature of 18°C."
    elif "london" in loc_lower:
        return "The current weather in London is Cloudy with a temperature of 15°C."
    else:
        return (
            f"The current weather in {location} is Clear with a temperature of 22°C."
        )


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    from_u = from_unit.upper().strip()
    to_u = to_unit.upper().strip()

    if from_u == to_u:
        return f"{value}°{to_u}"

    if from_u == "C" and to_u == "F":
        converted = (value * 9 / 5) + 32
        return f"{value}°C is equal to {converted:.1f}°F"
    elif from_u == "F" and to_u == "C":
        converted = (value - 32) * 5 / 9
        return f"{value}°F is equal to {converted:.1f}°C"
    else:
        return "Invalid units. Please specify 'C' or 'F'."


# Mapping for local execution
TOOL_MAP = {
    "calculator": calculator,
    "get_weather": get_weather,
    "convert_temperature": convert_temperature,
}

# Native JSON schemas for Groq API
GROQ_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a mathematical expression string (e.g., '145 * 12 / 4').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Valid mathematical string",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetches current weather and temperature (in Celsius) for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., Baku",
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
            "description": "Converts temperature between Celsius ('C') and Fahrenheit ('F').",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Temperature value to convert",
                    },
                    "from_unit": {"type": "string", "description": "C or F"},
                    "to_unit": {"type": "string", "description": "C or F"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]