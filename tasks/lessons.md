# Lessons Learned

- [x] **Parse before Repair**: When implementing JSON repair logic for LLM outputs, always attempt to parse the raw string first. Only apply repair transformations if the initial parse fails. This prevents "over-repairing" (e.g., double-escaping newlines) when the LLM actually provides valid JSON.
