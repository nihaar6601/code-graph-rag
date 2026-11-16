# Shop retrieval fixture

A small illustrative Java shop for Phase 5 retrieval comparisons. It has nine
ordinary named classes and 27 source-declared methods under `src/shop`. There
are no frameworks, generated entities, explicit constructors, or third-party
Java dependencies. The existing parser does not synthesize default constructors.

## Responsibilities and flow

- `OrderController` delegates placement, cancellation, and status requests.
- `OrderService` coordinates checkout and cancellation.
- `InventoryService` reserves, releases, and reports stock.
- `PaymentService` simulates charging and refunds with an aggregate cents ledger.
- `ShippingService` quotes delivery fees, dispatches, and retrieves tracking codes.
- `OrderRepository` stores order details and statuses in memory.
- `StockRepository` maintains per-SKU quantities, initially twenty units per SKU.
- `PriceCalculator` computes merchandise subtotal, SAVE10 discount, and total.
- `AuditLog` records and inspects per-instance activity history.

Checkout reserves stock, calculates merchandise and shipping charges, attempts
payment, saves an order, dispatches it, and records an event. Payment rejection
releases the reservation. Cancellation checks the stored status, refunds,
restocks, updates the status, and records an event. Unknown status/tracking
queries return explicit sentinel strings. Cross-file calls use explicit imports,
declared fields, and static utility calls. Java collection and String calls
remain external/unresolved in the graph; they are not shop dependencies.

## Benchmark contract

`../../benchmarks/shop_queries.json` contains 36 manually authored questions:
5 lookup, 4 implementation_only, 5 dependency, 5 behavior, 4 responsibility,
4 ambiguous, 5 multi_method, and 4 unrelated. Each record has exactly `question`,
`category`, and `expected`. Expected entries use the existing `entity_label`
format: qualified class names or qualified method signatures without spaces
between parameter types. Class questions expect class entities. Unrelated
questions expect empty lists.

Labels describe the intended answer scope, independently of retrieval rankings.
Implementation-only questions exclude dependencies; dependency questions include
the entry point and the explicitly requested path. Behavior questions select the
methods needed to explain that behavior, rather than every transitive callee.
Ambiguous questions deliberately name the plausible interpretations to make
labeling reproducible. Multi-method questions state the operations to retrieve.
External library methods and implicit constructors are never expected labels.

The existing indexer, graph builder, and `retrieval.entity_label` validate that
all 67 expected references resolve to exactly one entity (33 distinct labels).
Indexing yields 36 entities, 27 CONTAINS edges, and 25 resolved CALLS edges;
19 external invocation records remain unresolved. No retrieval ranking was used
to author or adjust these labels.

## Nonproduction behavior

This is illustrative, in-memory, nonproduction code. There is no HTTP server,
database, authentication, network payment, carrier integration, or durable audit
trail. Instances own their state, and all state is lost on restart. Payment
charges are limited to 100000 cents per charge; SAVE10 discounts round down in
integer cents. LOCAL delivery costs 500 cents and other zones cost 1500 cents.

Cancellation trusts caller-supplied SKU, quantity, and paid amount; it does not
verify these against saved order details or revoke a shipment. The ledger is
aggregate rather than per-order. Stock defaults accept any SKU. Arithmetic can
overflow, inputs are not comprehensively validated, and operations are neither
thread-safe nor transactional. These simplifications provide readable retrieval
examples, not a safe commerce implementation.
