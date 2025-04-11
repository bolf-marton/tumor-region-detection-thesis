from typing import Any

def print_success(message: Any) -> None:
    """Print a green success message with a checkmark icon."""
    print(f"\n\033[92m✅ {message}\033[0m")

def print_info(message: Any) -> None:
    """Print an info message with an info icon."""
    print(f"\nℹ️ {message}")

def print_warning(message: Any) -> None:
    """Print a yellow warning message with a warning icon."""
    print(f"\n\033[93m⚠️ {message}\033[0m")

def print_error(message: Any) -> None:
    """Print a red error message with an error icon."""
    print(f"\n\033[91m❌ {message}\033[0m")