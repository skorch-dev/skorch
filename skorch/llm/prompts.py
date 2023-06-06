"""Different prompts used for LLMs"""

DEFAULT_PROMPT_ZERO_SHOT = """You are a text classification assistant.

The text to classify:

```
{text}
```

Choose the label among the following possibilities with the highest probability:

{labels}

Only return the label, nothing more.

Your response:
"""

DEFAULT_PROMPT_FEW_SHOT = """You are a text classification assistant.

Choose the label among the following possibilities with the highest probability.
Only return the label, nothing more:

{labels}

Here are a few examples:

{examples}

The text to classify:

```
{text}
```

Your response:
"""
