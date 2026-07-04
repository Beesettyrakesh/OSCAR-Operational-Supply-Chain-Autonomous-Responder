# Security Rules

## Never access secrets

- Never read, open, search, or display any `.env` file.
- Never execute commands that print environment variables such as `env`, `printenv`, or `set`.
- Never ask for API keys, passwords, tokens, or secrets unless I explicitly request secret management.
- If a task appears to require reading `.env`, stop and ask for confirmation instead.
- Use placeholder values like YOUR_API_KEY in generated code.
