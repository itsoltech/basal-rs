using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Diagnostics.CodeAnalysis;

namespace ITSOL.Basal;

/// <summary>Base SDK exception. Original causes are retained in InnerException.</summary>
public class BasalException(string message, Exception? innerException = null) : Exception(message, innerException);

/// <summary>Non-2xx HTTP response. The body is intentionally absent from Message.</summary>
public sealed class BasalHttpException(int statusCode, JsonNode? body, string? retryAfter)
    : BasalException($"Basal HTTP {statusCode}")
{
    /// <summary>HTTP status code, including native validation 422 and overload 529.</summary>
    public int StatusCode { get; } = statusCode;
    /// <summary>JSON error body, or a JSON string holding a non-JSON response.</summary>
    public JsonNode? Body { get; } = body;
    /// <summary>Unmodified Retry-After header; no automatic retry.</summary>
    public string? RetryAfter { get; } = retryAfter;
}

/// <summary>Reusable asynchronous client for /v1/basal. No automatic retries.</summary>
public sealed class BasalClient : IDisposable
{
    private const string SynchronousFactoryRequired = "The answer factory must be synchronous. Await asynchronous work before calling DecideAsync.";
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
    private readonly HttpClient http;
    private readonly bool ownsHttp;
    private readonly Uri baseUrl;
    private readonly string? apiKey;
    private readonly TimeSpan timeout;
    private readonly int maxResponseBytes;

    /// <summary>Create a pooled client. baseUrl is the server root, optionally with a proxy prefix, without /v1.</summary>
    public BasalClient(string baseUrl = "http://127.0.0.1:8000", string? apiKey = null, TimeSpan? timeout = null,
        int maxResponseBytes = 8 * 1024 * 1024)
    {
        (this.baseUrl, this.apiKey, this.timeout) = Validate(baseUrl, apiKey, timeout);
        this.maxResponseBytes = ValidateLimit(maxResponseBytes);
        http = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false }) { Timeout = Timeout.InfiniteTimeSpan };
        ownsHttp = true;
    }

    /// <summary>Use an externally owned HttpClient, including IHttpClientFactory clients.
    /// Configure its handler with AllowAutoRedirect=false. Its own Timeout also applies.</summary>
    public BasalClient(HttpClient httpClient, string baseUrl, string? apiKey = null, TimeSpan? timeout = null,
        int maxResponseBytes = 8 * 1024 * 1024)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        (this.baseUrl, this.apiKey, this.timeout) = Validate(baseUrl, apiKey, timeout);
        this.maxResponseBytes = ValidateLimit(maxResponseBytes);
        http = httpClient;
    }

    private static (Uri Url, string? Key, TimeSpan Timeout) Validate(string baseUrl, string? apiKey, TimeSpan? timeout)
    {
        if (!Uri.TryCreate(baseUrl, UriKind.Absolute, out var uri) || (uri.Scheme != "http" && uri.Scheme != "https")
            || string.IsNullOrEmpty(uri.Host) || !string.IsNullOrEmpty(uri.UserInfo)
            || !string.IsNullOrEmpty(uri.Query) || !string.IsNullOrEmpty(uri.Fragment))
            throw new ArgumentException("Expected an HTTP(S) server URL without credentials, query or fragment", nameof(baseUrl));
        if (uri.Scheme != "https" && !uri.IsLoopback)
            throw new ArgumentException("baseUrl requires HTTPS or loopback HTTP (e.g. an SSH tunnel)", nameof(baseUrl));
        if (apiKey is not null && (apiKey.Length == 0 || apiKey.Contains('\r') || apiKey.Contains('\n')))
            throw new ArgumentException("Expected a nonempty, single-line token", nameof(apiKey));
        var duration = timeout ?? TimeSpan.FromSeconds(120);
        if (duration <= TimeSpan.Zero || duration.TotalMilliseconds > uint.MaxValue - 1)
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be positive and at most 4294967294 ms");
        return (new Uri(uri.AbsoluteUri.TrimEnd('/') + "/"), apiKey, duration);
    }

    private static int ValidateLimit(int limit)
        => limit > 0 ? limit : throw new ArgumentOutOfRangeException(nameof(limit), "Response byte limit must be positive");

    private async Task<T> SendAsync<T>(HttpMethod method, string path, object? body, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(method, new Uri(baseUrl, path));
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        if (apiKey is not null) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        if (body is not null) request.Content = new StringContent(JsonSerializer.Serialize(body, JsonOptions), Encoding.UTF8, "application/json");
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(timeout);
        HttpResponseMessage response;
        string raw;
        try
        {
            response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }
        catch (OperationCanceledException error) { throw new BasalException("Basal HTTP request timed out", error); }
        catch (HttpRequestException error) { throw new BasalException("Basal HTTP transport failed", error); }
        using (response)
        {
            try
            {
                if (response.RequestMessage?.RequestUri is Uri effectiveUri && effectiveUri != request.RequestUri)
                    throw new BasalException("Basal HTTP transport followed a redirect");
                using var stream = await response.Content.ReadAsStreamAsync(deadline.Token).ConfigureAwait(false);
                using var buffer = new MemoryStream();
                var chunk = new byte[8192];
                for (;;)
                {
                    var count = await stream.ReadAsync(chunk, deadline.Token).ConfigureAwait(false);
                    if (count == 0) break;
                    if (count > maxResponseBytes - buffer.Length)
                        throw new BasalException("Basal response exceeded the configured byte limit");
                    buffer.Write(chunk, 0, count);
                }
                raw = new UTF8Encoding(false, true).GetString(buffer.GetBuffer(), 0, checked((int)buffer.Length));
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }
            catch (OperationCanceledException error) { throw new BasalException("Basal HTTP request timed out", error); }
            catch (HttpRequestException error) { throw new BasalException("Basal HTTP transport failed", error); }
            catch (IOException error) { throw new BasalException("Basal HTTP response read failed", error); }
            catch (DecoderFallbackException error) { throw new BasalException("Invalid Basal response encoding", error); }
            if (!response.IsSuccessStatusCode)
            {
                JsonNode? errorBody;
                try { errorBody = JsonNode.Parse(raw); }
                catch (JsonException) { errorBody = JsonValue.Create(raw); }
                var retryAfter = response.Headers.TryGetValues("Retry-After", out var values) ? string.Join(", ", values) : null;
                throw new BasalHttpException((int)response.StatusCode, errorBody, retryAfter);
            }
            try { return JsonSerializer.Deserialize<T>(raw, JsonOptions) ?? throw new BasalException("Basal returned null JSON"); }
            catch (JsonException error) { throw new BasalException("Invalid Basal JSON response", error); }
        }
    }

    /// <summary>Decide from a non-null JSON state and typed question map.</summary>
    public async Task<TypedDecisionResponse> DecideAsync(JsonNode state, IReadOnlyDictionary<string, QuestionSpec> questions,
        DecisionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state); ArgumentNullException.ThrowIfNull(questions);
        if (questions.Count == 0) throw new ArgumentException("Questions must not be empty", nameof(questions));
        if (options?.Model is string model) Questions.Text(model, "model");
        if (options?.Facts is FactsMode facts && !Enum.IsDefined(facts)) throw new ArgumentException("Invalid facts");
        var snapshot = questions.ToArray();
        var wireQuestions = new JsonObject();
        foreach (var (id, question) in snapshot) { Questions.Text(id, "question ID"); ArgumentNullException.ThrowIfNull(question); wireQuestions[id] = question.ToJson(); }
        var request = new JsonObject { ["state"] = state.DeepClone(), ["questions"] = wireQuestions };
        if (options?.Model is string name) request["model"] = name;
        if (options?.Facts is FactsMode mode) request["facts"] = mode == FactsMode.Auto ? "auto" : "off";
        if(options?.Lang is string lang){if(lang is not ("pl" or "en"))throw new ArgumentException("Invalid lang");request["lang"]=lang;}
        if(options?.Detail is string detail){if(detail is not ("default" or "high"))throw new ArgumentException("Invalid detail");request["detail"]=detail;}
        if(options?.Media is { } media){
            if(media.Count is < 1 or > 8)throw new ArgumentException("Media requires 1 to 8 inputs");
            foreach(var item in media){ArgumentNullException.ThrowIfNull(item);if(item.Type is not ("image" or "audio"))throw new ArgumentException("Invalid media type");Questions.Text(item.Data,"media data");}
            request["media"]=JsonSerializer.SerializeToNode(media);
        }
        var response = await DecideAsync(request, cancellationToken).ConfigureAwait(false);
        return new TypedDecisionResponse(response, snapshot);
    }

    /// <summary>Build named answer fields using an anonymous object or a class. The factory runs once,
    /// before I/O. Read the fields after awaiting the response; they become available only after all
    /// answers pass validation. The builder is closed when the factory returns.</summary>
    public async Task<TypedDecisionResponse<TAnswers>> DecideAsync<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties | DynamicallyAccessedMemberTypes.PublicFields)] TAnswers>(JsonNode state,
        Func<QuestionBuilder, TAnswers> createAnswers, DecisionOptions? options = null,
        CancellationToken cancellationToken = default) where TAnswers : notnull
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(createAnswers);
        var builder = new QuestionBuilder();
        TAnswers answers;
        try { answers = createAnswers(builder); }
        finally { builder.Close(); }
        if (answers is null) throw new ArgumentException("The answer factory must return an object", nameof(createAnswers));
        builder.BindShape(answers);
        if (builder.Questions.Count == 0) throw new ArgumentException("The answer factory must add at least one question", nameof(createAnswers));
        var response = await DecideAsync(state, builder.Questions, options, cancellationToken).ConfigureAwait(false);
        builder.Complete(response);
        return new TypedDecisionResponse<TAnswers>(response, answers);
    }

    /// <summary>Asynchronous answer factories are unsupported. Prepare asynchronous inputs before calling DecideAsync.</summary>
    [Obsolete(SynchronousFactoryRequired, error: true)]
    public Task<TypedDecisionResponse<TAnswers>> DecideAsync<TAnswers>(JsonNode state,
        Func<QuestionBuilder, Task<TAnswers>> createAnswers, DecisionOptions? options = null,
        CancellationToken cancellationToken = default) where TAnswers : notnull
        => throw new ArgumentException(SynchronousFactoryRequired, nameof(createAnswers));

    /// <summary>Asynchronous answer factories are unsupported. Prepare asynchronous inputs before calling DecideAsync.</summary>
    [Obsolete(SynchronousFactoryRequired, error: true)]
    public Task<TypedDecisionResponse<TAnswers>> DecideAsync<TAnswers>(JsonNode state,
        Func<QuestionBuilder, ValueTask<TAnswers>> createAnswers, DecisionOptions? options = null,
        CancellationToken cancellationToken = default) where TAnswers : notnull
        => throw new ArgumentException(SynchronousFactoryRequired, nameof(createAnswers));

    /// <summary>Asynchronous answer factories are unsupported. The factory must return its answer shape synchronously.</summary>
    [Obsolete(SynchronousFactoryRequired, error: true)]
    public Task<TypedDecisionResponse> DecideAsync(JsonNode state,
        Func<QuestionBuilder, Task> createAnswers, DecisionOptions? options = null,
        CancellationToken cancellationToken = default)
        => throw new ArgumentException(SynchronousFactoryRequired, nameof(createAnswers));

    /// <summary>Asynchronous answer factories are unsupported. The factory must return its answer shape synchronously.</summary>
    [Obsolete(SynchronousFactoryRequired, error: true)]
    public Task<TypedDecisionResponse> DecideAsync(JsonNode state,
        Func<QuestionBuilder, ValueTask> createAnswers, DecisionOptions? options = null,
        CancellationToken cancellationToken = default)
        => throw new ArgumentException(SynchronousFactoryRequired, nameof(createAnswers));

    /// <summary>POST /v1/basal. Preserves native answer semantics.</summary>
    public Task<DecisionResponse> DecideAsync(DecisionRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return SendAsync<DecisionResponse>(HttpMethod.Post, "v1/basal", request, cancellationToken);
    }

    /// <summary>POST a native JSON object, including explicit null instruction values and future request fields.
    /// JsonObject preserves fields exactly; validation belongs to the server.</summary>
    public Task<DecisionResponse> DecideAsync(JsonObject request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return SendAsync<DecisionResponse>(HttpMethod.Post, "v1/basal", request, cancellationToken);
    }

    /// <summary>GET /v1/models.</summary>
    public Task<ModelsResponse> ModelsAsync(CancellationToken cancellationToken = default)
        => SendAsync<ModelsResponse>(HttpMethod.Get, "v1/models", null, cancellationToken);

    /// <summary>Dispose the owned HTTP client. Externally supplied clients remain open.</summary>
    public void Dispose() { if (ownsHttp) http.Dispose(); }
}
