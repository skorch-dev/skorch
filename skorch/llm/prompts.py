"""Different prompts used for LLMs"""


DELIM = "```"

DEFAULT_PROMPT_ZERO_SHOT = f"""You are a text classification assistant.

The text to classify:

{DELIM}
{{text}}
{DELIM}

Choose the label among the following possibilities with the highest probability.
Only return the label, nothing more:

{{labels}}

Your response:
"""

DEFAULT_PROMPT_FEW_SHOT = f"""You are a text classification assistant.

Choose the label among the following possibilities with the highest probability.
Only return the label, nothing more:

{{labels}}

Here are a few examples:

{{examples}}

The text to classify:

{DELIM}
{{text}}
{DELIM}

Your response:
"""
