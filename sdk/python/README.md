# basal-client (Python)

HTTP SDK for the native Basal API (`POST /v1/basal`). Python 3.11+; no GPU required on the client.

This package is versioned independently. Release tag: `sdk-python-vX.Y.Z`.
[Changelog](https://github.com/itsoltech/basal-rs/blob/main/sdk/python/CHANGELOG.md).

```sh
pip install basal-client
```

Choose either a normal dictionary or a declarative class. Both use the same factories.
A class gives each response field its exact inferred answer type:

```python
from basal_client import BasalClient, Questions, choice, noul

routing = choice("Do którego działu skierować zgłoszenie?", {"returns": "Zwroty", "support": "Wsparcie techniczne"})

class TicketQuestions(Questions):
    department = routing
    second_department = routing
    urgent = noul("Czy sprawa jest pilna?")

with BasalClient("http://127.0.0.1:8000") as client:
    response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", TicketQuestions)
    print(response.answers.department.choice)         # str, ChoiceAnswer
    print(response.answers.second_department.choice)  # independent answer
    print(response.answers.urgent.noul)               # float, NoulAnswer
```

Pass the class itself, without `()`. No response class, field annotations, casts, type-checker
plugin or code generator is needed. The same definition may appear under several names.

For dynamic question sets, use a dictionary and read answers by their request keys:

```python
from basal_client import BasalClient, choice

routing = choice("Which department?", {"returns": "Returns", "support": "Support"})
with BasalClient() as client:
    response = client.decide("The phone is broken.", {
        "department": routing,
        "second_department": routing,
    })
    print(response.answers["department"])
    print(response.answer("second_department"))
```

Dictionary values are checked answer dataclasses, with the static union `AnswerResult`.
Use `isinstance(answer, ChoiceAnswer)` to narrow that union when needed. Python cannot infer
an individual answer type from an arbitrary string key; the class form supplies that information.
Choice/label keys remain `str` in Python, rather than inferred literal unions.
`answer()` now takes a request key, not a question object (a change before the initial release).

Native asynchronous HTTP with connection pooling supports both forms:

```python
from basal_client import AsyncBasalClient, Questions, noul

class Availability(Questions):
    urgent = noul("Is this urgent?")

async def decide():
    async with AsyncBasalClient(timeout=120) as client:
        response = await client.decide("The service is unavailable.", Availability)
        return response.answers.urgent.noul
```

`models()` reads `/v1/models`. An optional `api_key` adds a bearer token
for an authenticating reverse proxy. The server root URL may include a proxy prefix; do not append `/v1`.
HTTPS is required except for loopback HTTP, suitable for local servers and SSH tunnels.
URL credentials/query/fragment are rejected. `trust_env=False` is the default: environment proxies and
certificate overrides are ignored. Set `trust_env=True` only when you intend to trust that configuration.
`max_response_bytes` defaults to 8 MiB and limits the actual decoded stream, including HTTP errors;
oversized responses raise `BasalResponseTooLargeError`. TLS verification remains enabled.
The client requests `Accept-Encoding: identity` and rejects compressed responses before decoding,
including when a proxy ignores that header. Configure proxies to return uncompressed JSON for this client.
Malformed or excessively nested JSON raises `BasalProtocolError`.
`timeout` is in seconds and applies to connect/read/write/pool phases, not a total wall-clock deadline.

`BasalHTTPError` provides `status_code`, `body` and the raw `retry_after` header. Transport failures raise
`BasalTransportError`; malformed successful JSON raises `BasalProtocolError`. Original exceptions are
available through `__cause__`. There are no automatic retries or redirects. Type hints describe the native
contract; successful responses are checked to be JSON objects, without full schema validation. Unknown
JSON fields are preserved. Server validation errors (422) and overload (529) remain distinct.

Supports `choice`, `noul`, `score`, `multi`, `act`, `facts` and `evidence`. Criterion order matters. Native
choice/score accepts 2–10 options; multi has independent probabilities and `set_confidence`. Evidence
offsets count Unicode characters. The SDK does not evaluate model decisions or perform selected actions.

Factories `choice`, `noul`, `score`, `multi`, `act` return concrete generic question classes.
Constructors `ChoiceQuestion(...)` etc. have the same signatures. Required response fields are
validated before returning; evidence defaults to an empty tuple. `response.raw` retains native
JSON and extensions. The one-argument `decide(request)` remains available for full native
JSON dictionaries. `model=` and `facts=` work with either two-argument form.

Schema fields may be inherited and overridden with the same question kind for compatibility
with both mypy and Pyright. Changing a field's kind or removing it with `None` is not supported
by mypy. Field order follows reversed Python MRO, like dataclasses: for `class D(M1, M2)`,
fields from `M2` precede those from `M1`; overriding a field retains its original position.
Names starting with `_` are reserved. Schema instances are populated by the SDK without
calling user constructors, and their fields cannot be reassigned. Define schemas declaratively;
do not mutate class definitions while requests are running. Dictionary question maps are also
snapshotted before I/O. Response dictionary keys are read-only; nested native data is not deeply frozen.

Typed response/answer containers do not support the copy/pickle protocols. For caching or
passing a result between processes, serialize `response.raw` as JSON. Use `copy.deepcopy(response.raw)`
when a detached native result is needed. Question definitions may also be stored in ordinary
classes, `ClassVar` attributes or dataclasses; outside `Questions` they remain question definitions.

[Full documentation and publication setup](https://github.com/itsoltech/basal-rs/blob/main/docs/SDK.md).

Runtime uses HTTPX with seven dependencies in the audited Python 3.11 graph. Building requires only the
hash-locked Flit Core backend; consumers of the wheel do not install the build backend.
