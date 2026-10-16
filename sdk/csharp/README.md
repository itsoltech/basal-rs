# ITSOL.Basal.Client

Asynchronous C# SDK for the native Basal API (`POST /v1/basal`). .NET 8+; no external runtime dependencies
or GPU requirement.
This package is versioned independently. Release tag: `sdk-csharp-vX.Y.Z`.
[Changelog](https://github.com/itsoltech/basal-rs/blob/main/sdk/csharp/CHANGELOG.md).

```sh
dotnet add package ITSOL.Basal.Client
```

```csharp
using ITSOL.Basal;

using var client = new BasalClient();
// These criteria may be loaded asynchronously from a database before preparation.
var criteria = new Dictionary<string, string> { ["returns"] = "Zwroty", ["support"] = "Wsparcie techniczne" };
var questions = QuestionBuilder.Prepare(q => new
{
    Department = q.Choice("Do którego działu skierować zgłoszenie?", criteria),
    Urgent = q.Noul("Czy sprawa jest pilna?"),
});
var response = await client.DecideAsync("Klient prosi o zwrot pieniędzy za uszkodzony telefon.",
    new { questions.Department, questions.Urgent });
Console.WriteLine($"Dział: {response.Answers.Department.Choice}, P(tak): {response.Answers.Urgent.Noul}");
```

Load database/enum values first, then call `QuestionBuilder.Prepare` once. Its factory is
synchronous and snapshots each built-in question's configuration. The prepared fields can be
cached and reused; they never receive answers. Each decision creates a fresh answer shape.
An asynchronous `Prepare` lambda is rejected at compile time and may report ambiguous overloads;
load its inputs before calling `Prepare`.

You can pass the whole anonymous prepared object, select a subset, rename a field, or reuse
one definition under several names:

```csharp
var one = await client.DecideAsync(state, new { Department = questions.Department });
var repeated = await client.DecideAsync(state, new
{
    FirstDepartment = questions.Department,
    SecondDepartment = questions.Department,
});
var whole = await client.DecideAsync(state, questions);
```

The two repeated members receive independent results. `response.Answers` has the same inferred
field types as the selected object, without casts or null suppression. Concurrent decisions
can share the prepared object. Reading a prepared field's result itself throws; read the
returned response instead. Fields from a previous response cannot be used as prepared definitions.

The direct-object `DecideAsync` is an extension in `ITSOL.Basal`; keep `using ITSOL.Basal` in scope.
It supports anonymous objects and named shapes with one public constructor whose parameter
names/types match all public answer members. Positional records are a convenient named form:

```csharp
public sealed record TicketQuestions(ChoiceField Department, NoulField Urgent);

// Inside the calling method:
// var prepared = QuestionBuilder.Prepare(q => new TicketQuestions(
//     q.Choice("Which department?", criteria), q.Noul("Is this urgent?")));
// var response = await client.DecideAsync(state, prepared);
// string choice = response.Answers.Department.Choice;
```

A named prepared class built with a builder-dependent constructor can select its members into
`new { questions.Department, questions.Urgent }` instead of being passed whole. Reconstruction
calls the matching public constructor before HTTP; it must preserve the supplied answer fields.
The SDK rejects constructors that swap or replace answer fields. Pass a concrete shape type;
a base/interface-typed variable is rejected instead of dropping derived questions.
Constructor metadata is shallow-copied and passed through the constructor again, so use direct
assignments rather than transformations. Members outside the constructor are not copied.
Answer getters must return stable fields. No private fields or readonly backing fields are modified.
The request order follows the selected shape's constructor parameters. Explicit wire IDs from
`q.Add` remain explicit; inferred IDs come from the selected member names. An explicit ID cannot
be repeated under multiple names. To cancel, use the named argument `cancellationToken: token`.

For questions prepared inline, the existing synchronous factory is also available:

```csharp
var response = await client.DecideAsync(state, q => new
{
    Department = q.Choice("Which department?", criteria),
    Urgent = q.Noul("Is this urgent?"),
});
```

The returned member name is the exact request ID: `Department` and `Urgent` above.
There is no separate key string. The anonymous object preserves each answer field's type.
Factories cover `Choice`, `Noul`, `Score`, `Multi`, `Act`, with the same typed options as
`Questions` factories. Read fields after awaiting
`DecideAsync`. The factory runs once, before HTTP, and the builder closes when it returns.
The factory must be synchronous: finish any asynchronous preparation before calling `DecideAsync`.
Factories returning `Task` or `ValueTask` are rejected at compile time.
An untyped `async` lambda can report ambiguous overloads; use a synchronous factory instead.
All results are validated before any field becomes readable. Failed or cancelled requests
leave captured fields unreadable; early reads throw `InvalidOperationException`.

Reuse a definition by creating one field for each member:

```csharp
var routing = Questions.Choice("Which department?", ("returns", "Returns"), ("support", "Support"));
var response = await client.DecideAsync(state, q => new
{
    Department = q.Choice(routing),
    SecondDepartment = q.Choice(routing),
});
// The two IDs have independent answers, even though the definition is shared.
```

A named class can express the same shape:

```csharp
sealed class TicketQuestions(QuestionBuilder q, ChoiceQuestion routing)
{
    public ChoiceField Department { get; } = q.Choice(routing);
    public ChoiceField SecondDepartment { get; } = q.Choice(routing);
    public NoulField Urgent { get; } = q.Noul("Is this urgent?");
}

// Inside the calling method:
// var response = await client.DecideAsync(state, q => new TicketQuestions(q, routing));
// string selected = response.Answers.Department.Choice;
```

Return inferred fields as initialized, readable public properties or public fields on the
factory's object; inference uses the compile-time returned type and examines only its top-level
answer members. Nested or `object`-typed members do not provide inferred IDs. Question order
follows field creation in the factory, independently of member-discovery order. Names preserve case.
Use an anonymous object, record, class or struct for inferred names; tuples with inferred fields
are rejected because tuple aliases are not available at runtime. Inherited base-class members
are included. Inherited interface members and hidden duplicate fields are not supported;
return a concrete shape with unambiguous members.
Answer property getters are read once before HTTP; do not construct fields inside a getter.

For inline DecideAsync factories, create a fresh object and its fields in each invocation. The SDK rejects fields from
another builder and rejects using one inferred field under two different member names before HTTP.
Reuse definitions through `q.Choice(routing)` etc. instead of caching inline-factory fields in `static` or with `??=`.
Fields created by `QuestionBuilder.Prepare` may be cached and shared as described above.
Do not use builders concurrently or add questions after the factory returns.
Fields are read-only views of checked result records; `.Result` exposes the record itself.
Member names are discovered with built-in .NET reflection; no source generator, dynamic compilation
or external dependency is added. To set a wire ID explicitly, `q.Add("custom_id", definition)` remains
available and overrides the member name; tuples using only explicit IDs also remain available.
Generic custom definitions can use `q.From(definition)` and the field's `.Result`.

For dynamic question sets, a dictionary remains available:

```csharp
var response = await client.DecideAsync("A damaged phone.",
    Questions.Create(("department", routing), ("second_department", routing)));
if (response.Answer("department") is ChoiceResult answer)
    Console.WriteLine(answer.Choice);
// response.Answers["second_department"] is also available.
```

Dictionary values have the common `AnswerResult` type and can be narrowed with pattern matching.
`Answer` now accepts a request ID, not a question object (changed before the initial release).
Use the object factory for exact named field types without pattern matching or null suppression.

For structured states, criteria and cost matrices use `System.Text.Json.Nodes.JsonObject`/`JsonArray`.
Supports `choice`, `noul`, `score`, `multi`, `act`, `facts` and `evidence`. Extra native fields are retained
with `JsonExtensionData`; do not duplicate explicit property names in `Extra`.
For a completely native request, use the `DecideAsync(JsonObject)` overload. It preserves explicit
JSON null fields, including `instructions: null`; null DTO properties are omitted.

`ModelsAsync()` reads `/v1/models`. All methods accept a `CancellationToken`.
The configurable timeout (default 120 seconds) includes response-body reading. `apiKey` optionally sends a
bearer token to an authenticating proxy. Use the server root (possibly with a proxy prefix), without `/v1`.

The default constructor owns its HTTP connection pool; dispose the SDK client when finished. The constructor
accepting an existing `HttpClient` leaves ownership to its caller and supports `IHttpClientFactory`. Configure
that client's handler with `AllowAutoRedirect=false`; its own timeout also applies. The owned handler
also sets `UseProxy=false`. An injected client controls its own proxy and TLS settings; keep certificate
verification enabled and configure the client's behavior before sending requests.

HTTPS is required except for loopback HTTP, suitable for a local server or SSH tunnel. URL credentials,
query and fragment are rejected. `maxResponseBytes` defaults to 8 MiB and limits the actual stream,
including HTTP errors. An oversized response raises `BasalException`.

`BasalHttpException` provides `StatusCode`, `Body` and raw `RetryAfter`. Other SDK failures raise
`BasalException` with the original cause in `InnerException`. Caller cancellation remains an
`OperationCanceledException`. No automatic retries. Responses deserialize into typed native DTOs.

Native choice/score accepts 2–10 options. Preserve criterion insertion order. Multi probabilities are
independent, with `SetConfidence`. Evidence offsets count Unicode characters, not UTF-16 units; prefer
`EvidenceSpan.Text` when displaying a quote. The SDK does not execute selected actions.

Namespace: `ITSOL.Basal`. Factories in `Questions` and concrete `ChoiceQuestion`, `NoulQuestion`,
`ScoreQuestion`, `MultiQuestion`, `ActQuestion` classes have typed options. `Create` builds the question
map without generic annotations. The object factory retains exact field types in
`response.Answers`; the dictionary overload returns checked records keyed by request ID.
`response.Raw` retains native DTOs/extensions. Add `DecisionOptions(Model: ..., Facts: ...)`
and a cancellation token to either the dictionary or object-factory overload; the raw DTO/JsonObject overloads remain.


[Full documentation and publication setup](https://github.com/itsoltech/basal-rs/blob/main/docs/SDK.md).
