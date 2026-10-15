"""The notebook is delivered with its outputs, so they must never contain personal data.

This scans every output of analysis.ipynb for things that look like an email, a phone
number, a birthdate column, a pseudonymization token or a customer ID.
Skipped if the notebook does not exist.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "analysis.ipynb"

PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?:\+?52[\s-]?)?\(?\d{2,3}\)?[\s-]?\d{3,4}[\s-]?\d{4}"),
    "token": re.compile(r"\b[0-9a-f]{16}\b"),
    "customer_id": re.compile(r"\bC\d{6}\b"),
    # A raw PII column shown as a table header (the word "email" in prose is fine).
    "pii_column": re.compile(
        r"<th>\s*(email|telefono|fecha_nacimiento|notas_crm|email_hash|phone_hash)\s*</th>"
    ),
}


def output_text(notebook: dict) -> str:
    chunks = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            chunks.append("".join(output.get("text", [])))
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/"):
                    chunks.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(chunks)


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not built")
@pytest.mark.parametrize("kind", list(PATTERNS))
def test_notebook_outputs_contain_no_personal_data(kind):
    text = output_text(json.loads(NOTEBOOK.read_text()))
    matches = PATTERNS[kind].findall(text)
    assert not matches, f"{kind} found in notebook outputs: {matches[:3]}"
