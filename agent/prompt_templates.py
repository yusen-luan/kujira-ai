"""Prompts + response parsing for the propose/repair LLM calls.

Output contract the LLM must follow (enforced by parse_response, not by
--json-schema, since we want full raw file contents rather than
JSON-escaped strings):

    HYPOTHESIS: <1-3 sentences>

    ```python:data.py
    <full new file content, only if changed>
    ```

    ```python:baseline.py
    <full new file content, only if changed>
    ```

Any file block whose name isn't in ALLOWED_FILES is dropped by the caller.
"""
import re

ALLOWED_FILES = ('data.py', 'baseline.py')

SYSTEM_PROMPT = """You are an ML engineering agent iterating on a recommendation-ranking \
baseline (KuaiRand-Pure, Factorization Machine).

You may only propose changes to two files: data.py and baseline.py. Anything else you \
write is ignored.

Hard constraints you must preserve, because a fixed harness calls into these files \
directly and cannot be changed:
- data.py must keep a `load(data_dir)` function returning the same split dict shape, \
and an `encode(splits)` function returning `(enc, dim)` where `enc[name] = (X, y, users)`.
- baseline.py must keep a `run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, \
patience=4, seed=0, verbose=True)` function returning `{'valid': {...}, 'test': {...}}` \
(each a dict with GAUC, nDCG@5, primary). Do not rename, reorder, or drop these keyword \
arguments.
- numpy and the Python standard library only. No new pip dependencies, no torch, no \
sklearn, no internet access.
- Don't try to change evaluate.py's scoring semantics — you won't be shown that file and \
can't edit it anyway.

Your job each turn: propose exactly ONE focused hypothesis for improving the validation \
`primary` metric (mean of GAUC and nDCG@5), then rewrite the full contents of whichever \
of data.py / baseline.py that hypothesis requires. Keep it to one attributable change per \
turn (one feature, one hyperparameter, one small architectural tweak) — not a grab-bag of \
unrelated changes, since we need to be able to tell what caused any metric change.

Output format, exactly:

HYPOTHESIS: <1-3 sentences: what you're changing and why you think it will help>

```python:data.py
<full new file content, only if you are changing this file>
```

```python:baseline.py
<full new file content, only if you are changing this file>
```

Omit a file's fenced block entirely if you are not changing it. Output nothing else \
outside this format — no preamble, no explanation after the code."""


def _format_history(history):
    if not history:
        return "(no iterations yet)"
    lines = []
    for h in history:
        status = h['status']
        if status == 'accepted':
            lines.append(f"iter {h['iter']}: ACCEPTED \"{h['hypothesis']}\" "
                         f"-> primary {h['primary']:.4f} (was {h['prev_best']:.4f})")
        elif status == 'rejected':
            lines.append(f"iter {h['iter']}: rejected (no improvement) \"{h['hypothesis']}\" "
                         f"-> primary {h['primary']:.4f} (best stays {h['prev_best']:.4f})")
        else:  # failed
            lines.append(f"iter {h['iter']}: FAILED after retries \"{h['hypothesis']}\" "
                         f"-> {h['error_summary']}")
    return "\n".join(lines)


def build_propose_prompt(best_code, history, best_primary):
    hist_txt = _format_history(history)
    return f"""Current best validation primary metric: {best_primary:.4f}

History of past iterations:
{hist_txt}

Current data.py:
```python
{best_code['data.py']}
```

Current baseline.py:
```python
{best_code['baseline.py']}
```

Propose your next hypothesis and code change now, following the output format from the \
system prompt."""


def build_repair_prompt(attempt_code, hypothesis, error_text):
    files_txt = "\n\n".join(
        f"Current {fname}:\n```python\n{content}\n```"
        for fname, content in attempt_code.items()
    )
    return f"""Your previous attempt for this hypothesis failed to run.

HYPOTHESIS: {hypothesis}

Error output:
```
{error_text[-4000:]}
```

{files_txt}

Fix the bug. Keep the same hypothesis text. Output the corrected full file(s) using the \
exact format from the system prompt (HYPOTHESIS: line + fenced code block(s))."""


_FENCE_RE = re.compile(r"```python:([^\n`]+)\n(.*?)```", re.DOTALL)
_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS:\s*(.*?)(?:\n```|\Z)", re.DOTALL)


def parse_response(text):
    """Returns (hypothesis: str, files: dict[filename -> content])."""
    files = {}
    for m in _FENCE_RE.finditer(text):
        fname = m.group(1).strip()
        content = m.group(2)
        if content.endswith("\n"):
            content = content[:-1]
        files[fname] = content
    hyp_match = _HYPOTHESIS_RE.search(text)
    hypothesis = hyp_match.group(1).strip() if hyp_match else ""
    return hypothesis, files
