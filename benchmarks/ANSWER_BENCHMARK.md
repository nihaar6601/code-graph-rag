# Shop answer benchmark

`shop_answers.json` is a hand-authored answer-quality companion to
`shop_queries.json`. Its 36 items preserve the original order, question,
category and expected retrieval labels verbatim. It contains 70 factual points
across 32 answerable queries and four unrelated queries requiring insufficient
evidence. It does not contain generated model answers or evaluation scores.

## Ground truth

References and facts were written by reading the nine Java classes under
`examples/shop/src/shop/`, not by inspecting retrieval results. Those files are
the authority for fixture behavior. In particular:

- PaymentService is a simulated, per-instance aggregate ledger. Charges accept
  1 through 100000 cents inclusive; refunds must be positive and cannot exceed
  the aggregate balance. This is not per-order payment verification.
- Unseen SKUs start with 20 demonstration units. Positive releases restore stock.
- SAVE10 divides the subtotal by 10 using Java integer division; for nonnegative
  subtotals the discount rounds down, not the final amount due.
- LOCAL shipping costs 500 cents; other zone values cost 1500 cents.
- Cancellation trusts caller-supplied SKU, quantity and refund amount. References
  do not imply additional validation, transaction guarantees or provider calls.

## Item schema and scoring contract

Each item has:

- `question`, `category`, `expected`: unchanged retrieval benchmark fields.
  `expected` is the original retrieval target set, not a requirement to cite
  every label in every factual claim.
- `expect_insufficient`: true only for the four unrelated queries.
- `reference_answer`: manually written concise explanation of fixture behavior.
  For implementation-only questions this is a behavior reference, not an exact
  code-output or response-format oracle.
- `facts`: one to four independently identified factual points for answerable
  queries. Each has a globally unique `id`, a human-readable `description`,
  alternative `patterns`, and qualified `sources` in the project's
  `entity_label` format (including parameter types for methods).

Compile patterns using Python `re.IGNORECASE` and apply `re.search` to each
**single generated claim text**. Any alternative matching that claim provides
lexical coverage for the fact. Do not concatenate separate claims to satisfy a
pattern. Keep the description as the intended meaning; regex alternatives
recognize common wording, Java identifiers and selected code expressions.
References can span multiple claims and are illustrative, not an exact string
match target. Pattern windows are bounded to reduce accidental distant matches.

`sources` identify the code entities supporting that particular fact. Some facts
about direct delegation are supported by the caller's method alone; the callee
need not be included merely to prove that the call occurs. Class responsibility
facts use class labels. A method label identifies that method's source range,
not every behavior of its class. A grader must separately check cited source
ranges and which source context was actually provided, including partial context
availability. A regex match alone must not establish citation correctness or
justify penalizing an answer for unavailable evidence. This file defines no
partial-context scoring formula or production grader behavior.

For unrelated questions, `facts` is empty, `expected` is empty and the reference
starts “Repository context does not contain evidence...”. This asks the system
to acknowledge insufficient supplied repository evidence; it is not a code fact
asserting global absence of a feature across arbitrary repositories.

`forbidden_patterns` is optional (`[{"pattern": "...", "description": "..."}]`).
This version deliberately omits it: bare matches such as “network charge” or
“negative refund” can incorrectly flag a correct negated statement. Clear fixture
falsehoods still require semantic review; there is no blanket keyword blacklist.

## Limitations

Lexical fact coverage is a proxy, not a complete judgment of semantic correctness,
faithfulness or hallucination. Correct paraphrases can miss the regexes; incorrect,
negated, conditional or quoted assertions can match. A multi-part fact may match
an alternative without establishing every clause. Numeric units, causality,
sequence and omitted qualifications need human review. Citation range validation
cannot prove that the cited code entails a generated claim.

The 36 questions concern a small synthetic in-memory Java shop. References are
not exhaustive specifications or general claims about production commerce.
Implementation-only formatting and code fidelity need separate evaluation.
Neither reference answers nor lexical matches establish successful end-to-end
LLM integration. Revisit the annotations when fixture code changes.

## Reproduce annotation validation

Run from the project root. This uses the existing parser and graph builder only;
no LLM calls, embeddings or retrieval results are needed.

```sh
.venv/bin/python - <<'PY'
import json
import re
from collections import Counter
from graph_builder import build_graph
from indexer import index_inputs
from retrieval import entity_label

queries = json.load(open('benchmarks/shop_queries.json'))
answers = json.load(open('benchmarks/shop_answers.json'))
graph = build_graph(index_inputs(['examples/shop/'])['files'])
labels = Counter(entity_label(graph, node) for node in graph.nodes)
assert len(answers) == len(queries) == 36
fact_ids = set()
pattern_count = 0
for query, answer in zip(queries, answers):
    for field in ('question', 'category', 'expected'):
        assert answer[field] == query[field]
    insufficient = query['category'] == 'unrelated'
    assert type(answer['expect_insufficient']) is bool
    assert answer['expect_insufficient'] == insufficient
    assert isinstance(answer['reference_answer'], str)
    assert answer['reference_answer'].strip()
    for label in answer['expected']:
        assert labels[label] == 1, label
    if insufficient:
        assert answer['facts'] == []
        assert answer['reference_answer'].startswith(
            'Repository context does not contain evidence')
    else:
        assert 1 <= len(answer['facts']) <= 4
    for fact in answer['facts']:
        assert isinstance(fact['id'], str) and fact['id']
        assert fact['id'] not in fact_ids
        fact_ids.add(fact['id'])
        assert isinstance(fact['description'], str) and fact['description']
        assert fact['sources'] and fact['patterns']
        for source in fact['sources']:
            assert labels[source] == 1, source
        compiled = []
        for pattern in fact['patterns']:
            assert isinstance(pattern, str) and pattern
            compiled.append(re.compile(pattern, re.IGNORECASE))
            pattern_count += 1
        # Annotation smoke check only; actual scoring is per generated claim.
        assert any(p.search(answer['reference_answer']) for p in compiled), fact['id']
    for forbidden in answer.get('forbidden_patterns', []):
        assert forbidden['description']
        pattern = re.compile(forbidden['pattern'], re.IGNORECASE)
        assert not pattern.search(answer['reference_answer'])
assert sum(a['expect_insufficient'] for a in answers) == 4
assert len(fact_ids) == 70
print(f'{len(answers)} queries; {len(fact_ids)} facts; {pattern_count} valid patterns')
PY
```

This checks annotation consistency, unique source labels, regex syntax and
reference coverage. It does not execute an answer-quality benchmark against a
model or validate generated citations.
