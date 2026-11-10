# Prompts

Every piece of text qagents itself puts in front of a model. Each one is a module
attribute you can read, log or replace, for example
`qagents.prompts.TOOL_FAILED = "..."`.

```python title="qagents/prompts.py"
--8<-- "src/qagents/prompts.py"
```
