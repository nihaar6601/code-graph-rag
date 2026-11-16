# Code Graph RAG: repository-wide Java code graph

Parse Java source with Tree-sitter, then build a NetworkX directed multigraph of
classes, methods, constructors, ownership, and straightforward method dependencies.
Repository indexing, local dense retrieval, lexical/hybrid ranking and graph-based
context retrieval are supported, with optional OpenAI-compatible LLM answering.
No graph/vector database or frontend is included. LLM requests are explicitly opt-in;
indexing, retrieval and the automated suite need no API keys or LLM network access.

## Install

Use Python 3.11 or newer. From this project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Direct runtime dependencies are Tree-sitter, its Java grammar, NetworkX, tiktoken,
NumPy, and Sentence Transformers (including its PyTorch/Transformers dependencies).
Tests use standard-library `unittest`.

## Run

```sh
# Index a directory recursively and print a concise summary.
.venv/bin/python main.py examples/
.venv/bin/python main.py examples/ --json /tmp/repository-graph.json

# Multi-file example demonstrating cross-file calls.
.venv/bin/python main.py examples/repository/ --json /tmp/payment-graph.json
.venv/bin/python main.py examples/repository/ --verbose

# Existing file-based commands remain supported and list entities/calls.
.venv/bin/python main.py examples/PaymentService.java
.venv/bin/python main.py examples/Calculator.java examples/PaymentService.java examples/NestedClasses.java
.venv/bin/python main.py
.venv/bin/python parser.py
```

With no arguments, `main.py` uses the original PaymentService example. `parser.py`
still runs the root-level Calculator demo. Directories and explicit file arguments
can be mixed. Explicit file arguments are honored even inside an ignored directory.

Directory mode recursively discovers `.java` files and excludes `.git`, `target`,
`build`, `.gradle`, `node_modules`, `.venv`, and `__pycache__`. Symlink files and
subdirectories are skipped during scanning to avoid cycles and indexing outside
the selected tree. Skipped paths and scan errors are reported in JSON.

The summary reports discovered/successful/error files, skipped entries, classes,
methods, constructors, node/edge totals, and resolved/unresolved calls. `--verbose`
also lists entities and individual invocations. Empty repositories produce an empty
graph with a zero-file summary. File or scan errors produce a nonzero exit status,
but valid files continue to be indexed and partial JSON can still be exported.
JSON export errors also return status 1; JSON output cannot replace an input path.

## Execution flow and files

1. `main.py` reads CLI arguments and calls `index_inputs()` in `indexer.py`.
2. `indexer.py` discovers files, deduplicates canonical input paths, sorts them,
   and calls `parse_file()` for each. In directory mode it selects a common root.
3. `parser.py` uses Java Tree-sitter fields to extract packages, imports, classes,
   declared fields, method signatures, constructors, parameters, lexical local
   variables, original source, locations, and method invocation records. It also
   collects syntax/encoding/file diagnostics without terminal output.
4. `graph_builder.py` adds entity nodes and `CONTAINS` edges. It asks `resolver.py`
   to resolve invocations using a repository symbol index, then adds supported
   `CALLS` edges. The original parsed dictionaries are not mutated.
5. `main.py` prints the results and optionally exports JSON.

Functions remain separately defined; parsing, discovery, resolution, and graph
construction are independent. Source and context are passed explicitly.

The original examples remain in `examples/`. The new `examples/repository/` has
three Java files under `src/demo/`: `app/PaymentController.java`,
`service/PaymentService.java`, and `util/Audit.java`.

## Cross-file resolution

For a call such as `service.processPayment()`, the resolver first identifies the
receiver's declared type from locals, parameters, or fields. Local bindings retain
lexical scopes and shadow fields/class names, even when their type is unknown.
`this.field.method()` is supported for a directly declared field.

Class lookup uses the current/enclosing named class, nested member type names,
explicit imports, same-package names, and fully qualified names. It never picks
an arbitrary class using only its simple name across the whole repository.
Package/import source and locations are retained in the exported file metadata.

Unqualified and `this.method()` calls search only the declaring class. A
class-qualified call such as `Audit.record()` requires a declared static method.
Calls with a known receiver type may target a declared method in another file.
Argument counts must match fixed-arity parameters. To preserve conservative
behavior, **all same-name overload sets remain unresolved**, even if argument
counts differ. Argument types are not inferred.

The repository example produces five resolved invocation edges:

- Controller field receiver → `PaymentService.processPayment()`.
- Controller local receiver → `PaymentService.processPayment()`.
- Controller parameter receiver → `PaymentService.processPayment()`.
- Controller static class receiver → `Audit.record()`.
- Service `this` receiver → `PaymentService.validatePayment()`.

`System.out.println(...)` remains an unresolved invocation. There are three files,
three classes, four methods, seven nodes, and nine edges (four `CONTAINS`, five
`CALLS`). No constructor is synthesized. The original standalone example's
`PaymentService() {}` is explicitly declared in its Java source.

## JSON and identities

The JSON contains:

- `files`: source identities, package/import metadata, status, and diagnostics.
- `entities`: class, method, and constructor nodes with source/locations and IDs.
- `relationships`: `source`, `target`, `kind` (`CONTAINS` or `CALLS`), and metadata.
- `invocations`: all extracted method calls, original receiver/text/location,
  `caller`, `resolved_target` (or `null`), and `unresolved_reason`.
- `indexing` (CLI exports): common root, discovered count, skipped paths, scan
  diagnostics. Individual file parse/read errors remain in `files`.

Directory-mode source paths and entity IDs use normalized POSIX paths relative to
the selected root. With multiple roots/outside files the base is their common
ancestor. Single/multi-file mode without a directory keeps absolute identities for
backward compatibility. The root itself remains absolute in indexing metadata.

IDs include file identity, qualified class name, kind, method name, and parameter
types. Same-named classes in different packages and overloaded declarations do not
merge. Local-class IDs include an enclosing-method source offset. Duplicate
source declarations receive source-offset suffixes. Repeated physical inputs are
deduplicated; deterministic file/AST order makes repeated indexing of unchanged
inputs produce identical JSON. This is a full in-memory rebuild, not a persistent
or incremental index. Offsets/absolute file mode are not edit/move-stable IDs.

Graph construction uses `nx.MultiDiGraph` to retain separate edges/locations for
repeated calls. `parse_source(text, file_path=...)` also supports in-memory sources;
give each distinct source a distinct file identity when combining results.
`build_graph(result_or_results)` and `graph_to_dict(graph)` can be used without
the CLI. Exported source positions use one-based lines and **byte columns**, with
an exclusive end position; byte offsets are zero-based and exclusive at the end.

## Conservative limits

This is declared-type dependency analysis, not a Java compiler or a prediction of
runtime dispatch. It does not validate argument types, definite assignment,
abstractness, all access rules, or all semantic errors. Private cross-class and
non-public cross-package method targets are conservatively left unresolved.

Unsupported/ambiguous receivers keep invocation records and reasons instead of
false graph edges. This includes wildcard/static-import resolution, external
libraries, overloads, varargs, generic receiver types/type parameters, arrays,
`var` inference, pattern-variable flow analysis, chained return values, arbitrary
field chains, inheritance lookup, `super`, reflection, dependency injection, and
dynamic dispatch. Static-context checks reject obvious instance calls. Calls from
malformed files or into files with diagnostics are not resolved.

Only named ordinary classes are entities. Anonymous classes, interfaces, enums,
records, annotation types, lambdas, method references, and initializer blocks are
not fully modeled. Their bodies do not leak calls or classes into an unrelated
owner. Calls inside lambdas/anonymous bodies are intentionally omitted. Local
classes are extracted but are not exported as globally resolvable repository types.
Object creation and explicit `this(...)`/`super(...)` constructor invocations are
not Java `method_invocation` nodes, so they do not create constructor-call edges.

Tree-sitter recovery reports `ERROR` and missing nodes; malformed files preserve
partial entities. Empty, unreadable, missing, and non-UTF-8 files have diagnostics.
Compiler-validity and complete Java symbol/type analysis remain future work.

## Test and reproduce

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py examples/repository/ --json /tmp/code-graph-rag-phase3.json
.venv/bin/python main.py examples/ --json /tmp/code-graph-rag-all-examples.json
```

The original 40 tests remain unchanged. Additional tests in
`tests/test_repository.py` cover recursive discovery, exclusions, unreadable files,
scan errors, symlinks, relative identities, imports, receiver bindings/shadowing,
static calls, duplicate class names, ambiguous/overloaded/external targets,
malformed dependencies, deterministic output, and repository CLI export.

## Phase 4: retrieve source context

Install the updated requirements and prepare the tokenizer once:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'from token_counter import get_tokenizer; print(get_tokenizer().name)'
```

`tiktoken` is the only added direct dependency. Its `cl100k_base` vocabulary is
fetched on first initialization and cached under `.cache/tiktoken/` (or the
`TIKTOKEN_CACHE_DIR` environment override). After setup, retrieval/evaluation works
offline; no LLM or external inference API is used. Copy the vocabulary cache when
moving to an offline environment.

```sh
.venv/bin/python main.py examples/repository/ \
  --query "How does payment processing work?" \
  --max-tokens 2000 --top-k 1 --depth 1 --output-dir output/demo

# Same lexical ranking without dependency traversal.
.venv/bin/python main.py examples/repository/ \
  --query "How does payment processing work?" --top-k 1 --no-expand

# Follow outgoing calls and optionally their incoming callers.
.venv/bin/python main.py examples/repository/ \
  --query "validatePayment" --top-k 1 --depth 2 --include-callers \
  --max-expanded 5 --context-json output/context.json
```

`--output-dir output/demo` creates the directory and writes `context.txt` and
`retrieval.json`. An explicit `--context-json` path requires an existing parent
directory. `--json` continues to export the **graph**, even with a query; use
`--context-json` for retrieval data. Output paths must be distinct and cannot
replace indexed input paths. Generated outputs and the tokenizer cache are ignored
by git. `--include-classes` explicitly allows complete class snippets; the default
retrieves methods/constructors without automatically including full classes/files.

### Retrieval flow

1. `retrieval.rank_entities()` tokenizes identifiers, camelCase, acronyms and
   underscores, removes a small English stopword list, and performs simple suffix
   normalization. Weighted method/class names, signatures, paths, return/parameter
   types and an inverse-document-frequency weight give a lexical score. Exact and
   partial identifier matches receive bonuses. Classes can be ranked, but default
   snippet candidates are methods/constructors, using owner names as metadata.
   Scores are relative heuristic relevance values, not probabilities.
2. The first `--top-k` candidates are considered as direct matches. There is no
   hidden whole-class source ranking or whole-repository text chunking.
3. Complete source snippets are checked and fitted first. Outgoing `CALLS` edges
   from accepted seeds are traversed breadth-first up to `--depth` (default 1).
   `--include-callers` adds incoming `CALLS` traversal; `--no-expand` or `--depth 0`
   disables graph expansion. `--max-expanded` (default 10) bounds dependency count.
   Unresolved calls cannot be followed because they have no graph target.
4. Cycles, repeated invocation edges and overlapping source ranges cannot duplicate
   snippets. Direct matches precede graph discoveries; dependencies receive a score
   halved at each hop from their discovery parent. Owner labels come from
   `CONTAINS` edges without inserting their complete source bodies.
5. `assemble_context()` formats the question and numbered snippets with entity ID,
   name, class attribution, file, line range, reason, score and original source.
   `retrieve_context()` returns this text plus structured snippets, candidates,
   omission diagnostics, counts, configuration, tokenizer name and latency.

In Python, call `retrieve_context(graph, question, ...)` on an indexed graph. Relative
source paths require `graph.graph['indexing']['root']` or an explicit `root=`.
Absolute-path single-file graphs work directly. Ranking and context are deterministic
for an unchanged graph/options; the measured `latency_ms` naturally varies.

### Source correctness and token budgets

Newly parsed file metadata includes a SHA-256 fingerprint of the complete indexed
bytes. Retrieval reads the source file, verifies that fingerprint, and validates
its byte span, original text, lines and byte columns. Missing, unreadable, modified,
malformed, or unverifiable sources are omitted with diagnostics. Even changes
outside a selected method require re-indexing. Older graph snapshots without
fingerprints must be re-indexed. Retrieval uses a per-request byte cache: snippets
are internally consistent with the snapshot read during that request, not a lock
against concurrent filesystem edits.

`token_counter.py` counts the **entire assembled context string** using tiktoken's
`cl100k_base`, including the question, headers, IDs, attribution, reasons, scores,
source and separators. Every proposed insertion is recounted before acceptance;
the final `token_count` never exceeds `--max-tokens`. JSON metadata outside the
`context` string is not intended as model context and is not counted. Chat-message
wrappers, system instructions, a future model's other tokenizer, and output tokens
are not included. This is a reproducible baseline tokenizer, not a claim about
all LLMs' billing or context limits.

Methods are never silently truncated. `oversized_method`/`oversized_class` reports
a snippet that cannot fit even alone; `budget_exhausted` reports one that does not
fit after higher-priority snippets. If the question/header exceeds the budget, the
returned context is empty with `question_over_budget`. Empty queries/invalid limits
are errors; unmatched queries return no snippets and a `no_matches` diagnostic.
The bounded traversal/ranking and whole-method policy can omit relevant code:
inspect the returned diagnostics rather than assuming completeness.

## Evaluate structure-only versus graph-enhanced retrieval

```sh
.venv/bin/python evaluate.py examples/repository/ \
  --benchmark benchmarks/payment_queries.json \
  --max-tokens 2000 --top-k 1 --depth 1 --json output/evaluation.json

.venv/bin/python -m unittest discover -s tests -v
```

`benchmarks/payment_queries.json` manually labels four questions with exact method
labels. `evaluate.py` runs the same graph, queries, top-k and budgets for both
strategies, changing only dependency expansion. The tokenizer is initialized and
warmed before timing. Each row reports retrieved entity IDs/names, total context
tokens, retrieval latency, direct/expanded counts, expected hits/misses, unexpected
entities, precision and recall. Indexing and export are excluded from latency;
ranking, source verification, expansion and context assembly are included. Timing
is a single local run, affected by warm filesystem caches and run order; this tiny
benchmark does not establish a latency advantage for either strategy.

Precision is expected hits divided by retrieved entities; recall is expected hits
divided by expected entities. Summary metrics are arithmetic means across questions
(macro averages). Empty predictions receive zero precision when expectations are
nonempty; both-empty precision is 1. Empty-ground-truth recall is 1. Missing or
ambiguous expected method labels fail evaluation rather than silently disappearing.

On the included benchmark at top-k 1/depth 1, structure-only macro precision/recall
is 1.000/0.6875; graph-enhanced is 0.875/0.9375. Payment-flow retrieval gains its
validation dependency. Controller retrieval gains the service and audit methods,
but misses validation until depth 2. For the question requesting *only* the
`processPayment` implementation, expansion adds an unwanted validation method:
precision falls from 1 to 0.5. Audit retrieval is unchanged. These examples show
both the benefit and cost of expansion; lower token consumption alone is not
better retrieval.

This is retrieval quality on a tiny hand-labeled example, **not final-answer
accuracy**. There is no LLM, semantic embedding, synonym model, learned reranker or
answer generator. Lexical matches can be weak or miss paraphrases; depth and
candidate caps, source availability and unresolved Java calls bound recall. A
broader independently labeled dataset is needed for general conclusions.

New Phase 4 files are `retrieval.py`, `token_counter.py`, `evaluate.py`,
`benchmarks/payment_queries.json`, and `tests/test_retrieval.py`. Parser/file metadata,
graph export and CLI integration were extended without replacing repository indexing.

## Phase 5: local embeddings, hybrid search and a traditional RAG baseline

### Model setup (explicit download)

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python embeddings.py --download-model
```

The default free model is `sentence-transformers/all-MiniLM-L6-v2`, running on CPU
with 384-dimensional vectors. No API key or paid service is needed. Model loading
is **local-only by default**. The separate setup command above, or the explicit
`--download-model` retrieval/evaluation flag, permits a Hugging Face model download.
No Java code or questions are sent to the model hub; inference is local. Model files
live in `.cache/models/` and vector files in `.cache/embeddings/`, excluded from git.
The default model occupies about **87 MiB** on disk in this environment (snapshot
symlinks refer to the same blobs). The full virtual environment with PyTorch and
other dependencies is about **1.1 GiB**. Other models can be much larger.

Use `--model`, `--model-cache`, `--cache-dir`, and optional `--revision` for another
local-compatible Sentence Transformers model or a pinned hub revision. Prepare the
same model/revision explicitly before offline use. The encoder uses token windows
at the model's own maximum sequence length and averages their normalized vectors,
then normalizes the result, so long code entities are not silently embedded only
from their beginning. The final source snippets always retain original source text.
The document/query pooling policy, actual loaded weights, vocabulary, model name,
revision and sequence limit identify the content-addressed vector cache. Cache
entries are non-pickle NumPy arrays; invalid/missing entries are regenerated.
Unchanged text can reuse vectors even if another file changed. Index/source ranges
are rebuilt, and snapshot verification still rejects stale code at retrieval time.

### Five strategies

```sh
.venv/bin/python main.py examples/repository/ \
  --query "How does payment processing work?" --strategy hybrid_graph \
  --max-tokens 2000 --top-k 3 --depth 1 --output-dir output/hybrid-demo

.venv/bin/python main.py examples/repository/ \
  --query "How does payment processing work?" --strategy traditional \
  --max-tokens 2000 --top-k 3

.venv/bin/python main.py examples/shop/ \
  --query "Show only the checkout implementation" --strategy graph \
  --top-k 1 --output-dir output/implementation-only
```

- `traditional`: fixed-size token-window chunks, dense embeddings and NumPy cosine
  similarity. Source discovery is shared with the repository scanner, but the
  retrieval index does **not parse Java, build entities or traverse a graph**.
  Even malformed-but-readable Java is chunked, as in a conventional text baseline.
- `structure`: existing lexical entity ranking, without graph expansion.
- `graph`: lexical entity seeds with query-aware graph expansion.
- `hybrid`: lexical and dense entity rankings fused with RRF; no graph expansion.
- `hybrid_graph`: the same fused seeds with query-aware graph expansion.

All use the same final context formatter/token budget and source verification.
Traditional retrieval only builds a graph if `--json` explicitly requests an
additional graph export; that graph is not used in its ranking. `--json` still
means graph export, `--context-json` means retrieval export, and `--output-dir`
saves `context.txt` plus `retrieval.json`. Old commands with **no `--strategy`**
keep Phase 4 behavior, including unconditional expansion unless `--no-expand` was
specified. Explicit strategy selection enables the Phase 5 policy. The old
`evaluate.py` invocation without `--strategies` also keeps its original two-strategy,
four-query evaluation for backward compatibility.

`chunking.py` uses cl100k_base windows, default `--chunk-size 128` and
`--chunk-overlap 24`. These token limits apply to code, not the final metadata;
complete chunk insertions are still checked against the final context budget.
Chunk byte offsets and inclusive line ranges describe exact original bytes. UTF-8
boundaries are respected: if an individual Unicode code point requires more tokens
than an unusually tiny chunk size, that indivisible character may exceed the code
chunk target. No character is dropped between windows. Overlapping chunks remain
separate conventional baseline snippets, so their repeated text consumes context
budget; evaluation unions overlap rather than double-counting evidence.

Semantic entity documents contain the qualified name, signature and original class
or method source. The default context policy stays method-oriented; explicit class
names, class/responsibility wording, or `--include-classes` allow complete class
snippets. These are query-text rules, not benchmark-label hints. The independent
`semantic_rank()` function is available for semantic entity searches.

Hybrid fusion is **equal-weight Reciprocal Rank Fusion**:
`score(entity) = sum(1 / (rrf_k + rank))`, ranks starting at 1. Defaults are
`--rrf-k 60`, `--lexical-candidates 20`, and `--semantic-candidates 20`.
IDs deduplicate candidates; metadata records each contributing rank and original
score. RRF scores are not calibrated semantic probabilities. `--min-similarity`
(default 0.2) filters dense cosine results; zero vectors never match. Semantic
nearest neighbors are not guaranteed relevant, especially for unrelated questions.
If semantic setup fails, hybrid returns an explicit `lexical_fallback` status and
diagnostic. Traditional returns `unavailable` and a nonzero CLI status rather than
pretending lexical results were dense retrieval. Full evaluation reports semantic
availability and returns nonzero if the model/index is unavailable.

### Query-aware expansion

`hybrid.py` implements four deterministic intents: `implementation_only`, `lookup`,
`dependency`, and `behavior`. Only/just/implementation or explicit exclusion of
related dependencies suppresses automatic expansion. Call/chain/dependency/trace
wording enables it; how/why/flow/behavior/explanation/comparison wording generally
enables it. Plain direct name lookups do not expand. These are lightweight English
rules, not semantic intent understanding or an LLM classifier.

For `graph` and `hybrid_graph`, `--expansion auto` is the default. Use
`--expansion always` to force traversal, or `--expansion never` / `--no-expand` to
prevent it (`--no-expand` takes precedence). `structure`, `hybrid`, and `traditional`
never expand regardless of that override. Existing depth/caller/entity-count caps
still apply, and unresolved invocations never become invented edges. A direct
implementation query can still return several direct matches when `--top-k` is
larger than one; set top-k 1 to request a single seed.

### Expanded evaluation and comparison table

`examples/shop/` is a separate nine-class, 27-method illustrative repository with
controller, service, repository and utility interactions. The original three-file
example and all earlier tests are preserved. `benchmarks/shop_queries.json` has
**36 manually labeled questions** across lookup, implementation-only, dependency,
behavior, responsibility, ambiguous, multi-method and unrelated categories. Four
questions have empty ground truth. Its labels were written from intended source
behavior, validated for existence/uniqueness, and not derived from retrieval output.
See `examples/shop/README.md` for fixture behavior and nonproduction limitations.

```sh
.venv/bin/python evaluate.py examples/shop/ \
  --strategies traditional,structure,graph,hybrid,hybrid_graph \
  --benchmark benchmarks/shop_queries.json --max-tokens 2000 --top-k 3 --depth 1 \
  --json output/phase5-evaluation.json

.venv/bin/python -m unittest discover -s tests -v
```

The evaluator prints a comparison table and writes it beside the JSON as
`output/phase5-evaluation.comparison.txt`. The JSON includes all 180 per-query rows,
retrieved IDs/source locations, intent/availability, snippet counts, expanded counts,
expected-entity coverage, precision/recall/F1, Recall@k, tokens, latency, model
identity/revision, installed package versions, and improvement/regression examples.
All strategies receive identical questions, top-k, token budgets and depth. Class
eligibility rules are identical across entity strategies.

**Common evaluation units matter:** primary precision/recall/F1 use **UTF-8 source
bytes**, not “one chunk equals one method.” Manually expected methods/classes map to
their exact source ranges; optional `expected_ranges` can directly specify
`file_path`, `start_byte` and exclusive `end_byte`. Retrieved and expected intervals
are each unioned per file before intersection. Precision is relevant retrieved bytes
divided by all retrieved source bytes; recall is covered expected bytes divided by
all expected bytes. F1 is their harmonic mean. This penalizes a large chunk's
irrelevant code and gives a partial chunk only partial recall. The same rule also
penalizes unnecessarily broad entity/class snippets. Attribution/formatting counts
against the token budget but is not labeled relevant source.

`relevant_entities` counts expected labels with at least 80% source-byte coverage;
`relevant_snippets` counts snippets touching expected bytes. These auxiliary counts
are **not** substituted for primary precision/recall. `recall_at_k` is source-byte
recall from the first k ranked candidates **before budget filtering and graph
expansion**; final context recall may differ. Zero-expected queries have recall 1;
precision is 1 only if nothing was returned. Metrics are macro means over all 36
queries, including unrelated ones. Source bytes weight formatting and long bodies;
this is a reproducible evidence-coverage proxy, not a universal semantic relevance
metric. The older Phase 4 evaluation retains its original entity-count metric and
must not be compared numerically as though it used these byte-based rules.

`overhead` separates graph indexing, chunk indexing, model loading, per-corpus vector
cache hits/misses and generation/persistence time. Setup is outside per-query
latency; query embedding, ranking/fusion, source verification, traversal and context
assembly are inside it. One encoder is shared across the compared strategies;
query embeddings are not reused across strategies. Traditional-only evaluation
builds a graph solely to locate expected annotation ranges, reported separately as
`annotation_only_graph_ms`; retrieval remains independent. Cold and warm cache runs
are different costs, so retain the first report separately when comparing overhead.
Cache writes and repeated model loading are not free even when no document vector
needs regeneration. Timings are single local runs with warm CPU/filesystem effects,
not a robust throughput or speed comparison.

### Phase 5 limits

This phase uses an English/general-purpose MiniLM, not a code-specialized model or
learned reranker. Whole-document window averaging can dilute a long class's meaning.
Dense similarity thresholds and RRF candidate counts are engineering choices, not
calibrated confidence. Lexical and semantic signals can add irrelevant code; graph
expansion inherits the existing conservative Java resolution limits and may hurt
precision. The benchmark is small, hand-labeled and synthetic, so results do not
establish superiority on real repositories or answer accuracy. No LLM is integrated.

New production modules are `embeddings.py`, `chunking.py`, `hybrid.py`, `strategies.py`
and `relevance.py`. `indexer.py` now exposes shared syntax-independent discovery;
`retrieval.py` accepts externally ranked seeds; the existing CLI/evaluator were
extended. Embedding and integration tests use deterministic fake encoders and do
not load or download real models. Real-model smoke tests/evaluation are explicit
commands, separate from the fast automated suite.

### Recorded Phase 5 validation

A local MiniLM run on the 36 shop queries at top-k 3, depth 1 and 2000 tokens yielded
macro source-byte precision/recall/F1:

- Traditional: 0.3938 / 0.8469 / 0.4885, about 432 context tokens per question.
- Structure: 0.6677 / 0.7212 / 0.6634, about 264 tokens.
- Graph: 0.6647 / 0.8068 / 0.7010, about 347 tokens.
- Hybrid: 0.6544 / 0.8103 / 0.6642, about 330 tokens.
- Hybrid graph: 0.6452 / 0.8744 / 0.6886, about 416 tokens.

Graph had the highest F1 in this run; hybrid graph had the highest recall. This is
not evidence that hybrid is universally better. For example, “Which class keeps
the per-instance activity history?” improves from structure F1 0 to hybrid F1 1,
while the exact shipment-tracking lookup drops from structure F1 1 to hybrid F1
about 0.204 because additional irrelevant code is retrieved. “Follow a stock
release...” improves from structure F1 about 0.511 to hybrid F1 1. Query-aware
expansion suppresses traversal for the implementation-only questions, but cannot
remove irrelevant **direct** candidates when top-k is larger than one.

The first recorded cache-miss report is `output/phase5-cold-evaluation.json`:
about 14 ms indexing, 4.46 s model loading, and 1.12 s vector generation plus
persistence (18 chunk and 36 entity vectors). A subsequent warm report reused all
54 vectors with zero regeneration; model startup still took several seconds.
`output/phase5-evaluation.json` and its `.comparison.txt` sidecar hold the final
query-level metrics and timings. `output/phase5-demos/` contains real, offline CLI
outputs for all five strategies, not mocked semantic results. These generated files
are local validation artifacts, excluded from git; regenerate them with the commands
above. Exact timings vary with hardware, concurrent work and filesystem caches.

## Phase 6: optional grounded question answering

`llm.py` is an independent OpenAI-compatible **Chat Completions** client using the
Python standard library, so Phase 6 adds no dependencies. `answering.py` connects
any existing retrieval strategy to a consistent evidence-only prompt. Answering
is disabled unless you pass `--answer`; the separate answer evaluator also requires
explicit opt-in for real calls. Existing retrieval/indexing commands and
`evaluate.py` remain retrieval-only.

### Complete offline mock demonstration

```sh
.venv/bin/python main.py examples/shop/ \
  --query "How does checkout handle a rejected payment?" \
  --strategy graph --optimization relevance --top-k 3 --max-tokens 2000 \
  --answer --provider mock --mock-response examples/mock_checkout_answer.json \
  --output-dir output/phase6-mock-demo
```

This uses a **hand-written fixed response**, not a model. It demonstrates the
retrieval → prompt → provider → parsed claims → validated citations → rendered
answer → metrics/export flow. It explains payment rejection, positive-quantity
stock restoration, and the charge amount guard with file/line citations. It is
not evidence of an LLM's accuracy, cost savings, or reliable grounding.

`--output-dir` additionally saves `answer.txt` and `answer.json` when answering.
`--answer-json path.json` exports just the complete answer result; its parent must
exist. `--context-json` and `--json` retain their retrieval and graph meanings.
Output paths cannot replace indexed Java or mock-response inputs. The JSON retains
retrieval evidence, prompt messages/version/hash, raw provider text, normalized
claims/citations, rejected claims, diagnostics, settings and metrics for audit.

### Configure a real provider

Configure API keys in environment variables or your shell's secret-management
workflow; never commit keys or pass their values as CLI options. Default key names
are `OPENAI_API_KEY` for `--provider openai`, `GROQ_API_KEY` for `groq`, and
`LLM_API_KEY` for `compatible`. `--api-key-env` names an alternate variable.
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL` and `LLM_API_KEY_ENV` are optional
configuration defaults. No keys, dotenv files or credentials are auto-discovered
from repository files.

```sh
# After securely configuring OPENAI_API_KEY and choosing your accessible model:
.venv/bin/python main.py examples/shop/ \
  --query "How does checkout handle a rejected payment?" \
  --strategy hybrid_graph --optimization relevance --answer \
  --provider openai --llm-model YOUR_MODEL \
  --max-tokens 2000 --max-output-tokens 512 --temperature 0 \
  --output-dir output/real-answer

# An already running local compatible server; no API key required:
.venv/bin/python main.py examples/shop/ --query "Explain checkout" \
  --strategy graph --answer --provider local \
  --base-url http://localhost:8000/v1 --llm-model YOUR_LOCAL_MODEL
```

Provider presets select the OpenAI or Groq API base; `--provider compatible`
requires a base URL (`--base-url` or `LLM_BASE_URL`). HTTP is permitted only for a
loopback `local` provider; remote endpoints use HTTPS, redirects are not followed,
and URL-embedded credentials are rejected. The compatible endpoint must support
non-streaming `/chat/completions` with `choices[].message.content`. No inference
server or local generative model is installed/launched by this project.

With **`--answer`**, `--model` is a shorthand for the LLM, matching the Phase 6
example command. Prefer unambiguous `--llm-model` and `--embedding-model`. Without
`--answer`, the old `--model` option still configures retrieval embeddings.
The default embedding model remains MiniLM regardless of the answer model.

The OpenAI and Groq presets send `max_completion_tokens`; generic/local presets send
`max_tokens`. Override with `--output-token-parameter` when your endpoint requires
another supported choice. `--json-mode` requests `response_format=json_object`;
the prompt always requires JSON, but the wire option is off by default for broad
compatibility. `--omit-temperature` supports models that reject a temperature
parameter. These choices follow the official [Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
There is no automatic model substitution or per-strategy prompt/parameter change.

`--request-timeout` defaults to 30 seconds per HTTP attempt. `--retries` defaults
to 1, with bounded exponential backoff for connection errors/timeouts and transient
408/429/5xx statuses. Authentication/request errors are not retried. Errors are
sanitized: no authorization headers, key values, provider error bodies, or source
payloads are put in error messages. Logs/config exports contain the key variable
name, never its value. A retry may duplicate billable work after an ambiguous
network timeout; successful-response usage alone cannot account for that.

### Troubleshooting Groq transport

On this macOS environment, Python's default OpenSSL CA store was empty even
though curl could verify Groq using macOS SecureTransport. The previous generic
connection/timeout message hid the certificate verification failure (code 20).
The client now loads certifi when default trust is empty, preserving explicit
CA configuration and certificate/hostname verification. Install dependencies
with `.venv/bin/python -m pip install -r requirements.txt`.

Inspect local TLS/proxy configuration without network access:

```bash
.venv/bin/python llm_transport.py
```

Explicitly check DNS and direct IPv4/IPv6 TCP/TLS connectivity without making an
HTTP request or sending API credentials or prompts:

```bash
.venv/bin/python llm_transport.py --host api.groq.com --tls-probe
```

These probes do not exercise a configured proxy. Proxy diagnostics report presence
only. The client preserves urllib proxy handling and accepts `--ca-bundle`,
`SSL_CERT_FILE`, or `SSL_CERT_DIR` for explicit trust configuration. Curl/requests
CA environment variables are not automatically applied to urllib.

Errors distinguish DNS, certificate verification, TLS negotiation, connection
timeout, read timeout, HTTP status failures, and malformed responses without
including provider error bodies or request contents. Retries retain their existing
policy. Optional `--connect-timeout 10 --read-timeout 60` separates connection setup
from socket reads; these are per-operation timeouts, not curl's total `--max-time`
deadline. Without these options, `--request-timeout` continues to govern both.

Groq now defaults to `max_completion_tokens`, matching the supplied curl request
and the [Groq API reference](https://console.groq.com/docs/api-reference).
Mocked tests verify the curl-equivalent request body and transport failures:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Prompt, citations and insufficiency

All five strategies share exactly the same versioned system instructions and JSON
evidence envelope. Retrieval scores/strategy names, expected facts, reference
answers and grading labels are **not** sent as instructions or hints. Every source
gets a stable-within-answer ID (`S1`, `S2`, ...), entity ID, filename, byte/line
range and original text. Source code/comments/strings are explicitly untrusted
data, not instructions. There is no code execution or LLM tool calling. This is
instruction hardening, not a guarantee against every model prompt-injection failure.

The provider must return JSON with `claims`, `insufficient_context` and
`uncertainties`. Each factual claim cites source IDs and file line ranges. The
application checks IDs and line bounds, renders valid claims with normalized
file/line citations, and excludes claims with invalid/missing attribution from the
rendered answer while retaining them in the audit JSON. Unverified assumptions
belong in explicitly labeled uncertainties. Valid citation **syntax/ranges do not
prove that code semantically supports a claim**; models can still misread code or
make false claims with valid-looking references.

No verified snippets means a deterministic insufficient-context answer, **no LLM
request**, and known zero request cost. Otherwise the model is instructed to give
supported partial claims and acknowledge missing evidence. Refusals, malformed
JSON/schema, invalid citations and output-limit truncation remain distinguishable
statuses rather than silently counted as successful grounded answers. Source
verification uses the existing indexed snapshot; it does not lock files against
edits after retrieval.

### Actual versus estimated tokens, cost and timing

Each answer records retrieved-context tokens; actual provider input/output/total
usage where reported; separate local estimates; retrieval/generation/end-to-end
latency; configured/requested and returned model; attempts; and cost estimate.
The provider's `prompt_tokens`, `completion_tokens` and `total_tokens` stay **null**
when unavailable—mock responses never manufacture actual usage. Partial usage stays
partial. Counts refer to the raw generated JSON, not just the shorter rendered
answer. Completion usage can include non-visible reasoning tokens, as described
in the official [token counting guide](https://developers.openai.com/api/docs/guides/token-counting).

Fallback estimates use cl100k_base visible text and approximate chat framing;
they are not model-specific billing counts and cannot estimate hidden reasoning.
The retrieved `--max-tokens` budget is not a full provider context-window limit:
the prompt adds system instructions and JSON attribution. Real reported input
usage therefore need not equal the retrieved context count.

Pricing is intentionally not hardcoded. Configure **both**
`--input-price-per-million` and `--output-price-per-million` to estimate USD from
reported counts, falling back to labeled estimates only when needed. Without
pricing, cost is unknown (`null`), not zero. Flat-rate estimates ignore cached-token
discounts, provider tiers and other billing details. Failed/ambiguous requests have
unknown usage/cost unless the provider reports it; retries are explicitly marked
as possibly incomplete accounting. Mock/no-request cost zero means no API was
called, not that real inference is free.

Setup/embedding/indexing overhead is separate from per-query timings. Generation
latency includes network, provider processing and retries. End-to-end includes
retrieval, prompt/answer processing and generation, excluding setup and file export.
`model_load_ms` in setup refers to the **retrieval embedding model**, not an API's
server-side generative-model startup, which the client cannot independently measure.

### Optional relevance-aware dependency experiment

`--optimization relevance` applies only to `graph`/`hybrid_graph`; default `none`
preserves Phase 5 expansion. It prioritizes query-matching seeds, ranks existing
CALLS dependencies by identifier/receiver overlap and enclosing if-branch evidence,
and filters zero-relevance dependencies. Failure-path questions can prioritize
calls inside a matching failure branch. Traversal remains depth/count bounded;
unresolved invocations still do not create edges. The optimizer reads query text,
source and graph metadata only—never benchmark files or expected facts.

On the checkout-rejection example, baseline graph retrieval includes ten snippets
and 1373 context tokens. Relevance mode selects checkout, charge and stock release:
three snippets and 491 tokens, excluding shipping, audit and persistence methods.
This is **retrieval/context reduction**, not demonstrated answer-quality or API-cost
improvement. The lexical/branch heuristics can discard needed code on other
questions; retain `--optimization none` as the experimental control. Existing
`--expansion`/`--no-expand` options still control whether expansion occurs at all.

### Separate answer-quality benchmark

`benchmarks/shop_answers.json` adds manually authored reference explanations,
70 factual points with regex alternatives, and relevant source labels for all
36 original shop questions. The four unrelated questions expect acknowledgment of
insufficient supplied evidence. It is separate from retrieval ranking. See
`benchmarks/ANSWER_BENCHMARK.md` for annotation provenance and limits.

```sh
# No LLM network calls: validate all five QA paths, using cached local embeddings.
.venv/bin/python evaluate_answers.py examples/shop/ --provider mock \
  --strategies traditional,structure,graph,hybrid,hybrid_graph \
  --repeats 2 --top-k 3 --max-tokens 2000 \
  --json output/phase6-mock-comparison.json

# Small opt-in REAL smoke test after configuring credentials and a supported model:
.venv/bin/python evaluate_answers.py examples/shop/ --answer \
  --provider openai --llm-model YOUR_MODEL --strategies graph \
  --limit 1 --max-requests 1 --retries 0 --max-output-tokens 256 \
  --json output/phase6-real-smoke.json

# Full REAL comparison: 36 questions x 5 strategies x 2 repeats = up to 360 calls.
.venv/bin/python evaluate_answers.py examples/shop/ --answer \
  --provider openai --llm-model YOUR_MODEL \
  --strategies traditional,structure,graph,hybrid,hybrid_graph \
  --max-tokens 2000 --max-output-tokens 512 --top-k 3 --depth 1 \
  --repeats 2 --max-requests 360 --retries 0 \
  --json output/phase6-real-comparison.json
```

`--max-requests` checks the upper bound of logical trials before calling the
provider; retries can add HTTP attempts. `--limit` selects a smaller prefix, not
a cheaper but biased per-strategy subset. Choosing a limit below planned trials
fails rather than silently producing an unbalanced comparison. Pricing/cost flags
are optional; token caps and request limits bound experiments but are not a hard
USD invoice guarantee. No real evaluation is automatically triggered by tests.

The evaluator reuses one prepared repository/model state and one provider/config
across strategies, rotates strategy ordering per question/repeat, and records all
query results plus aggregate means, missing-usage sample counts, failures,
insufficiency rates, and repeated-question variation. The `.comparison.txt` sidecar
shows a compact table; full JSON includes per-fact and per-citation diagnostics,
source fingerprints and the prompt/benchmark hashes. Index/model setup is reported
separately. Use the same settings and `--optimization none` versus `relevance` in
separate runs for a controlled optimization comparison.

The initial **deterministic proxy grader** reports:

- `correctness_proxy`: fraction of rendered claims matching at least one manually
  expected fact and not matching a manually configured known-false pattern.
- `completeness`: expected factual points lexically covered by rendered claims.
- `grounded_completeness`: facts additionally attributed to overlapping expected
  source entities/ranges. This is attribution, not semantic entailment.
- `available_fact_completeness`: coverage conditional on facts whose required source
  entities are each at least 80% present by source-byte coverage. This helps separate
  missing retrieval evidence from omitted answers; the threshold is only a proxy.
- Citation validity, known contradiction/rejected-claim counts, unverified-claim
  counts, and whether insufficiency was acknowledged when expected evidence was
  missing. Unmatched claims are **not automatically labeled hallucinations**.

A fact pattern must match a single claim; unrelated claims are not concatenated to
manufacture a match. Full-benchmark completeness remains visible even when retrieval
missed evidence. Unrelated/no-fact questions receive full credit only for an
insufficient-evidence response without factual claims. Factual paraphrases, negation,
causality and novel valid claims need human review: these proxies do not establish
semantic answer accuracy or detect all hallucinations. There is no LLM-based judge
in this phase, and retrieval precision/recall are never substituted for answer scores.

### Phase 6 validation artifacts and limitations

`output/phase6-mock-demo/answer.txt` and `.json` demonstrate a cited fixed answer.
`output/phase6-mock-comparison.json` exercises 360 query/strategy/repeat paths and is
labeled `mock_plumbing_only`: its scores are **not real-model quality comparisons**.
All actual provider token counts remain null. No API credentials were configured
in the implementation environment, so no real LLM smoke test, real answer-quality
comparison or claimed quality improvement was performed. Real evaluation remains
available through the explicit commands above.

New modules: `llm.py`, `answering.py`, `answer_grading.py`, `evaluate_answers.py` and
`context_optimizer.py`. The parser records branch context for optional prioritization;
retrieval accepts dependency priorities; the existing CLI exposes answering while
preserving earlier commands. New tests mock every external LLM call and require no
API keys, LLM network connection or paid inference. Run everything with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Phase 6.1: controlled real-provider validation workflow

The answer evaluator now supports a **credential-free dry run**, model-catalog
preflight, request and client-side cost caps, and static Markdown reports with
per-question human-review checklists. No new retrieval architecture, database,
frontend or dependency was added. The existing MockProvider is unchanged.

### Plan before any API requests

```sh
.venv/bin/python evaluate_answers.py examples/shop/ \
  --provider groq --llm-model YOUR_FREE_TIER_MODEL \
  --profile initial --checkout-smoke --compare-optimization \
  --retries 0 --max-requests 49 --max-api-requests 50 \
  --max-tokens 2000 --max-output-tokens 512 \
  --dry-run --json output/phase61-plan.json
```

Choose a model actually enabled for your provider/account, preferably a free-tier
or low-cost model for the initial run. `YOUR_FREE_TIER_MODEL` is a placeholder,
**not** an available-model or pricing claim. Dry-run validates configuration syntax
without an API key, provider instantiation, retrieval, tokenizer initialization,
embedding/model loading, downloads, or network calls. `--dry-run` takes precedence
even if `--answer` is also present. It displays provider/model, credential presence
(boolean only), selected questions/strategies, component request counts, retries,
context/output limits, approximate tokens, and cost when user-supplied pricing
exists. It cannot confirm model availability, quota, parameter support or pricing
remotely and says so explicitly.

`--profile initial` selects eight **existing, unchanged** answer annotations via
`benchmarks/phase61_initial.json`: shipment tracking lookup, implementation-only
checkout, pricing callees, order-status delegation, rejected-payment rollback,
coupon rounding, cancellation dependencies, and charge/refund/balance operations.
These cover direct lookup, cross-file/multi-step flows, failures, multiple methods
and no-expansion questions. Default `--profile all` preserves the earlier full
36-question evaluation; `--limit` can reduce either selection.

Optional stages are separate from the main comparison:

- `--checkout-smoke` runs “How does checkout handle a rejected payment?” first for
  every selected strategy. It reuses the existing rejected-payment facts unchanged
  and records their original annotation question for provenance. A smoke transport,
  schema, citation, refusal, truncation or budget failure stops later stages, with
  the partial smoke report retained. An insufficiency response is reported, not
  automatically certified as a correct answer.
- `--compare-optimization` runs the same checkout question with `none` and
  `relevance` for the selected graph-based strategies. Both runs share provider,
  model, prompt, budget, top-k and generation settings. They are not pooled into
  main benchmark averages. Per-pair differences include context tokens, actual
  input tokens where both are reported, and separately labeled proxy-grade deltas.
  Independently verified answer-quality differences stay unknown until reviewed.

With all five strategies and one repetition, the example plans **40 main answers,
5 smoke answers, 4 optimization answers: 49 completion requests**, plus one catalog
GET. With `--retries 0`, the upper HTTP count is 50; retries increase the planned
upper count. No-context trials may skip generation, so planned counts are upper
bounds, not claims about actual usage. Additional repetitions require `--repeats`.

### Live provider preflight and explicit authorization

Real execution still requires `--answer`, configured credentials in environment
variables, and a real model name. No automatic credit purchases, account creation,
or live inference occurs during implementation/testing. Keys are never printed or
stored in plans/reports; only public settings and the variable name/presence are
recorded. Before completions, the evaluator performs an authenticated, read-only
`GET /models` and checks the selected ID. Catalog errors/missing models stop
before generation. This checks advertised availability, **not** model quota or
support for every request setting. `--skip-model-check` is an explicit escape
hatch for compatible servers without a catalog; reports mark this unverified.

Configure both rates from your provider's current pricing, or explicitly zero if
that accurately describes your account/model. No rates are fetched or hardcoded.
For example, after securely configuring `GROQ_API_KEY`, `LLM_MODEL` and your own
`INPUT_USD_PER_MILLION` / `OUTPUT_USD_PER_MILLION` shell values:

```sh
# First inspect the priced plan. Set MAX_RUN_USD to your chosen small budget.
.venv/bin/python evaluate_answers.py examples/shop/ \
  --provider groq --llm-model "$LLM_MODEL" \
  --profile initial --checkout-smoke --compare-optimization --retries 0 \
  --max-requests 49 --max-api-requests 50 \
  --input-price-per-million "$INPUT_USD_PER_MILLION" \
  --output-price-per-million "$OUTPUT_USD_PER_MILLION" \
  --max-total-cost "$MAX_RUN_USD" \
  --dry-run --json output/phase61-priced-plan.json

# Explicit live opt-in; same settings and no more than one repetition by default.
.venv/bin/python evaluate_answers.py examples/shop/ --answer \
  --provider groq --llm-model "$LLM_MODEL" \
  --profile initial --checkout-smoke --compare-optimization --retries 0 \
  --max-requests 49 --max-api-requests 50 \
  --max-tokens 2000 --max-output-tokens 512 --temperature 0 \
  --input-price-per-million "$INPUT_USD_PER_MILLION" \
  --output-price-per-million "$OUTPUT_USD_PER_MILLION" \
  --max-total-cost "$MAX_RUN_USD" \
  --json output/phase61-live.json
```

Use `--provider openai` with `OPENAI_API_KEY` and an accessible model for OpenAI,
or the existing `compatible`/`local` configuration. With no pricing and no cost
cap, cost remains unknown and earlier commands still work. Setting a cost cap
requires explicit rates; free-tier availability must not be inferred just from a
provider name. `--max-requests` retains the legacy logical-trial cap and now covers
all requested stages; `--max-api-requests` separately bounds HTTP attempts,
including catalog preflight and retry reservations.

### Cost accounting and honest bounds

`evaluation_controls.py` wraps providers without changing mocks. Before each
completion it reserves a conservative amount for the **actual UTF-8 prompt bytes**,
chat-framing headroom, configured output limit and all allowed retries. Requests
that cannot fit remaining cost/request reservations are blocked before I/O. The
full lightweight plan is also checked before live execution; if its estimated
upper reservation exceeds the configured cap, execution is refused. A dry run can
still show an over-budget plan for inspection.

Successful reported input/output usage settles the reservation at configured
rates, releasing unused allowance. Unknown failed-attempt usage retains its
reservation. Missing/partial billable usage **under an active cost cap stops further
calls** rather than silently treating estimates as measured usage. Unknown prices
remain unknown in ledgers as well as answer metrics. Provider-reported partial
usage remains partial; explicit `provider_usage_reported` and
`provider_usage_complete` flags help audit coverage.

These are **conservative client-side controls, not a guaranteed provider invoice
ceiling**. Providers can have hidden framing, different tokenizers, non-token
charges, incorrect reported usage or different rates. If reported cost exceeds a
reservation/cap, the client stops subsequent requests but cannot undo that charge.
Catalog GETs count toward request limits but are not priced by completion-token
rates. Use a provider-side hard quota for an enforceable account-level ceiling.
The ledger is for this sequential evaluator, not a concurrent cross-process wallet.

Budget stops have their own `budget_exhausted` status, zero request cost for calls
that never happened, unknown actual tokens, and a `budget_stop_rate`. Partial runs
are labeled `incomplete_budget_do_not_rank_strategies` and return nonzero; they must
not be interpreted as completed fair quality comparisons. JSON preserves individual
results, observed costs, conservative reservations and stop reasons.

### JSON, Markdown and independent human review

`--json output/run.json` writes detailed JSON, `output/run.comparison.txt`, and
**`output/run.md`**. `--markdown path.md` chooses a separate report path, with or
without JSON. Output collisions and source/annotation overwrites are checked before
API requests. Failed model preflight is also exported with no fabricated answers.

The static Markdown report contains:

- Public provider settings, prompt/benchmark hashes in JSON, plan, setup and budget.
- A comparison table with proxy correctness/completeness, source grounding,
  citation validity, context tokens, **actual input/output usage versus estimates**,
  cost, latency, failure, insufficiency and budget-stop rates.
- Every query's reference answer and factual points, original retrieved context,
  exact evidence/source locations, generated answer, citations and grading details.
- Unchecked human-review items for factual correctness, completeness, semantic
  grounding, unsupported claims and appropriate insufficiency. Every run is marked
  **not independently reviewed**; a valid citation is not proof of entailment.
- Separate smoke and optimization sections, including unknown actual-token deltas
  when either side lacks provider usage. Smaller context is not scored as a better
  answer merely because it is smaller.

Proxy metrics remain the Phase 6 deterministic rules, not semantic accuracy.
Unknown usage/cost stays unknown. Setup/indexing/cache/model-startup costs remain
separate from per-query retrieval and generation latency. Repeated-run variation is
available through explicit `--repeats`; one initial run does not establish a general
speed/quality/cost advantage.

### Offline validation and current status

```sh
.venv/bin/python evaluate_answers.py examples/shop/ --provider mock \
  --profile initial --checkout-smoke --compare-optimization --retries 0 \
  --max-requests 49 --max-api-requests 50 --max-total-cost 0 \
  --json output/phase61-mock-validation.json

.venv/bin/python -m unittest discover -s tests -v
```

The mock workflow uses cached local retrieval embeddings, makes **zero LLM HTTP
requests**, and validates 49 trial paths. It is plumbing validation, not a live
quality benchmark. Existing fixed mock answering still works unchanged. The saved
mock optimization reports show graph context 1373 → 491 tokens and hybrid graph
1376 → 493; actual LLM input-token and independently validated answer-quality
changes remain unknown.

No live credentials/model configuration were available in the implementation
environment, and no paid requests were authorized or made. Artifacts
`output/phase61-dry-run.json`/`.md` and `output/phase61-mock-validation.json`/`.md`
are explicitly dry/mock. A real initial benchmark and independent answer review
remain to be performed with the exact opt-in commands above. No real-model accuracy
or API-cost improvement is claimed.
