"""Task/domain labels from the user instruction and UltraFeedback source.

Labels are a property of the prompt, not the model response, so the same
prompt receives one domain even when several models answered it.

Priority is code, then writing, then reasoning (FLAN chain-of-thought items),
then question answering. Anything else is "other" and is dropped.
"""

import re

# Programming requests. Bare words like "function" or "code" are intentionally
# omitted because they fire on non-code prose ("dress code", "function of").
CODE_RE = re.compile(
    r"("
    r"```"
    r"|\b(python|javascript|typescript|java|c\+\+|c#|sql|html|css|php|ruby|rust|golang|perl)\b"
    r"|\b(bash|shell)\s+script\b"
    r"|\b(leetcode|source code|code snippet)\b"
    r"|\bprogramming\s+(problem|language|task|exercise)\b"
    r"|\bwrite\s+(a|an|me\s+a|me\s+an)\s+(program|function|script|class|method)\b"
    r"|\bimplement\s+(a|the|an)\s+(function|class|program|algorithm|script)\b"
    r"|\bdebug\s+(this|the|my)\s+code\b"
    r")",
    re.IGNORECASE,
)

# Requests to produce prose. Checked after code so "write a python script" stays code.
WRITING_RE = re.compile(
    r"("
    r"\b(short story|poem|essay|blog(?:\s+post)?|fiction|novel|lyrics|song lyrics|"
    r"cover letter|e-mail|email|screenplay)\b"
    r"|\b(compose|draft|rewrite)\s+(a|an|me\b|the\b)"
    r"|\bwrite\s+(?:me\s+)?(?:a|an|the)?\s*(?:\d[\d,\-]*\s*word[s]?\s+)?"
    r"(story|essay|poem|email|e-mail|article|letter|script|speech|dialogue|"
    r"summary|blog|review|paragraph|report|scene|caption)"
    r"|\bstory\s+about\b"
    r")",
    re.IGNORECASE,
)

QA_RE = re.compile(
    r"(\?|\b(what|who|why|how|when|where|which|explain|describe|define)\b)",
    re.IGNORECASE,
)


def label_domain(source: str, instruction: str) -> str:
    text = instruction or ""
    if CODE_RE.search(text):
        return "code"
    if WRITING_RE.search(text):
        return "writing"
    if source == "flan_v2_cot":
        return "reasoning"
    if source in {"truthful_qa", "false_qa"} or QA_RE.search(text):
        return "qa"
    return "other"
