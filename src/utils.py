import re
import os

def joules2kwh(data):
    """Converts energy in joules to kilowatt-hours."""

    return data / 3.6e6

def kwh2joules(data):
    """Converts energy in kilowatt-hours to joules."""

    return data * 3.6e6

def clean_filename(filename):
    # Define a safe filename pattern: replace unsafe characters with underscores
    safe_filename = re.sub(r'[\/:*?"<>|]', '_', filename)  # Replace unsafe characters

    # Strip leading/trailing whitespace
    safe_filename = safe_filename.strip()

    # Optionally: Truncate the filename to a safe length (if it's too long)
    return safe_filename
