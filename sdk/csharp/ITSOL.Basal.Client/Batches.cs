using System.Globalization;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace ITSOL.Basal;

/// <summary>Expiry of an uploaded file or of batch result files: seconds after the anchor
/// (the server accepts anchor <c>created_at</c> and 3600–2592000 seconds).</summary>
public sealed class ExpiresAfter
{
    /// <summary>Expiry anchor; the server accepts only <c>created_at</c>.</summary>
    [JsonPropertyName("anchor")] public string Anchor { get; init; } = "created_at";
    /// <summary>Lifetime in seconds after the anchor.</summary>
    [JsonPropertyName("seconds")] public required long Seconds { get; init; }
    /// <summary>Additional fields. Do not repeat named fields above.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>A stored file: an uploaded input (purpose <c>batch</c>) or a batch result (<c>batch_output</c>).</summary>
public sealed class FileObject
{
    /// <summary>File ID (<c>file_</c> + 32 hex digits).</summary>
    [JsonPropertyName("id")] public required string Id { get; init; }
    /// <summary>Always <c>file</c>.</summary>
    [JsonPropertyName("object")] public string? Object { get; init; }
    /// <summary>Content length in bytes.</summary>
    [JsonPropertyName("bytes")] public long Bytes { get; init; }
    /// <summary>Creation time, Unix seconds.</summary>
    [JsonPropertyName("created_at")] public long CreatedAt { get; init; }
    /// <summary>Expiry time, Unix seconds, or null.</summary>
    [JsonPropertyName("expires_at")] public long? ExpiresAt { get; init; }
    /// <summary>Name given at upload, or the server's result file name. Never a path.</summary>
    [JsonPropertyName("filename")] public string? Filename { get; init; }
    /// <summary><c>batch</c> or <c>batch_output</c>.</summary>
    [JsonPropertyName("purpose")] public string? Purpose { get; init; }
    /// <summary>Always <c>processed</c> (OpenAI compatibility).</summary>
    [JsonPropertyName("status")] public string? Status { get; init; }
    /// <summary>Always null (OpenAI compatibility).</summary>
    [JsonPropertyName("status_details")] public string? StatusDetails { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Answer of DELETE /v1/files/{file_id}.</summary>
public sealed class DeletedFile
{
    /// <summary>Deleted file ID.</summary>
    [JsonPropertyName("id")] public required string Id { get; init; }
    /// <summary>Always <c>file</c>.</summary>
    [JsonPropertyName("object")] public string? Object { get; init; }
    /// <summary>Always true.</summary>
    [JsonPropertyName("deleted")] public bool Deleted { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>One page of a list, newest first. Pass <see cref="LastId"/> as <c>after</c> for the next page
/// while <see cref="HasMore"/> is true.</summary>
public sealed class ListPage<T>
{
    /// <summary>Always <c>list</c>.</summary>
    [JsonPropertyName("object")] public string? Object { get; init; }
    /// <summary>Page items.</summary>
    [JsonPropertyName("data")] public List<T> Data { get; init; } = [];
    /// <summary>ID of the first item, or null for an empty page.</summary>
    [JsonPropertyName("first_id")] public string? FirstId { get; init; }
    /// <summary>ID of the last item, or null for an empty page.</summary>
    [JsonPropertyName("last_id")] public string? LastId { get; init; }
    /// <summary>Whether further items follow <see cref="LastId"/>.</summary>
    [JsonPropertyName("has_more")] public bool HasMore { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Body of POST /v1/batches.</summary>
public sealed class CreateBatchRequest
{
    /// <summary>ID of an uploaded JSONL input file (purpose <c>batch</c>).</summary>
    [JsonPropertyName("input_file_id")] public required string InputFileId { get; init; }
    /// <summary>Endpoint of every record; the server accepts only <c>/v1/basal</c>.</summary>
    [JsonPropertyName("endpoint")] public string Endpoint { get; init; } = "/v1/basal";
    /// <summary>The server accepts only <c>24h</c>.</summary>
    [JsonPropertyName("completion_window")] public string CompletionWindow { get; init; } = "24h";
    /// <summary>Up to 16 string pairs (keys 1–64, values up to 512 characters), checked by the server.</summary>
    [JsonPropertyName("metadata")] public IReadOnlyDictionary<string, string>? Metadata { get; init; }
    /// <summary>Lifetime of each result file after its creation; omitted uses the server default.</summary>
    [JsonPropertyName("output_expires_after")] public ExpiresAfter? OutputExpiresAfter { get; init; }
    /// <summary>Additional request fields. The current server refuses unknown fields.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Record counts of a batch.</summary>
public sealed class BatchRequestCounts
{
    /// <summary>Number of records; 0 until the input is validated.</summary>
    [JsonPropertyName("total")] public long Total { get; init; }
    /// <summary>Records answered with status 200.</summary>
    [JsonPropertyName("completed")] public long Completed { get; init; }
    /// <summary>Other records, including expired ones.</summary>
    [JsonPropertyName("failed")] public long Failed { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>One input validation issue or the reason a batch failed.</summary>
public sealed class BatchErrorEntry
{
    /// <summary>Stable error code, e.g. <c>invalid_json_line</c> or <c>duplicate_custom_id</c>.</summary>
    [JsonPropertyName("code")] public required string Code { get; init; }
    /// <summary>Message of at most 256 bytes.</summary>
    [JsonPropertyName("message")] public string? Message { get; init; }
    /// <summary>Related parameter, or null.</summary>
    [JsonPropertyName("param")] public string? Param { get; init; }
    /// <summary>1-based line of the input file, or null.</summary>
    [JsonPropertyName("line")] public long? Line { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary><c>errors</c> of a batch.</summary>
public sealed class BatchErrors
{
    /// <summary>Always <c>list</c>.</summary>
    [JsonPropertyName("object")] public string? Object { get; init; }
    /// <summary>Recorded issues.</summary>
    [JsonPropertyName("data")] public List<BatchErrorEntry> Data { get; init; } = [];
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Batch status values. <see cref="Batch.Status"/> stays a string so newer values remain readable.</summary>
public static class BatchStatus
{
    /// <summary>The input file is being validated.</summary>
    public const string Validating = "validating";
    /// <summary>Validation failed; see <see cref="Batch.Errors"/>.</summary>
    public const string Failed = "failed";
    /// <summary>Validated; records run or wait for the GPU.</summary>
    public const string InProgress = "in_progress";
    /// <summary>Result files are being written.</summary>
    public const string Finalizing = "finalizing";
    /// <summary>Finished; result files exist.</summary>
    public const string Completed = "completed";
    /// <summary>The 24h window ended; unfinished records are failed with <c>batch_expired</c>.</summary>
    public const string Expired = "expired";
    /// <summary>Cancellation requested.</summary>
    public const string Cancelling = "cancelling";
    /// <summary>Cancelled; result files exist for finished records.</summary>
    public const string Cancelled = "cancelled";

    /// <summary>Whether the status is final (failed, completed, expired or cancelled).</summary>
    public static bool IsTerminal(string? status)
        => status is Failed or Completed or Expired or Cancelled;
}

/// <summary>A batch resource. Times are Unix seconds; times of states not reached are null.</summary>
public sealed class Batch
{
    /// <summary>Batch ID (<c>batch_</c> + 32 hex digits).</summary>
    [JsonPropertyName("id")] public required string Id { get; init; }
    /// <summary>Always <c>batch</c>.</summary>
    [JsonPropertyName("object")] public string? Object { get; init; }
    /// <summary>Always <c>/v1/basal</c>.</summary>
    [JsonPropertyName("endpoint")] public string? Endpoint { get; init; }
    /// <summary>Effective model of the records once resolved.</summary>
    [JsonPropertyName("model")] public string? Model { get; init; }
    /// <summary>Validation issues or the failure reason, or null.</summary>
    [JsonPropertyName("errors")] public BatchErrors? Errors { get; init; }
    /// <summary>Input file ID.</summary>
    [JsonPropertyName("input_file_id")] public required string InputFileId { get; init; }
    /// <summary>Always <c>24h</c>.</summary>
    [JsonPropertyName("completion_window")] public string? CompletionWindow { get; init; }
    /// <summary>One of <see cref="BatchStatus"/>; kept as a string for newer values.</summary>
    [JsonPropertyName("status")] public required string Status { get; init; }
    /// <summary>File of status-200 result lines, or null.</summary>
    [JsonPropertyName("output_file_id")] public string? OutputFileId { get; init; }
    /// <summary>File of other result lines, or null.</summary>
    [JsonPropertyName("error_file_id")] public string? ErrorFileId { get; init; }
    /// <summary>Creation time.</summary>
    [JsonPropertyName("created_at")] public long CreatedAt { get; init; }
    /// <summary>Start of execution.</summary>
    [JsonPropertyName("in_progress_at")] public long? InProgressAt { get; init; }
    /// <summary>End of the 24h inference window (export may finish later).</summary>
    [JsonPropertyName("expires_at")] public long? ExpiresAt { get; init; }
    /// <summary>Start of finalization.</summary>
    [JsonPropertyName("finalizing_at")] public long? FinalizingAt { get; init; }
    /// <summary>Completion time.</summary>
    [JsonPropertyName("completed_at")] public long? CompletedAt { get; init; }
    /// <summary>Failure time.</summary>
    [JsonPropertyName("failed_at")] public long? FailedAt { get; init; }
    /// <summary>Expiry time.</summary>
    [JsonPropertyName("expired_at")] public long? ExpiredAt { get; init; }
    /// <summary>Cancellation request time.</summary>
    [JsonPropertyName("cancelling_at")] public long? CancellingAt { get; init; }
    /// <summary>Cancellation time.</summary>
    [JsonPropertyName("cancelled_at")] public long? CancelledAt { get; init; }
    /// <summary>Record counts.</summary>
    [JsonPropertyName("request_counts")] public BatchRequestCounts? RequestCounts { get; init; }
    /// <summary>Metadata given at creation, or null.</summary>
    [JsonPropertyName("metadata")] public Dictionary<string, string>? Metadata { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>One line of a JSONL input file. Serialize each line with System.Text.Json and join with LF.</summary>
public sealed class BatchInputLine
{
    /// <summary>1–256 bytes, unique in the file byte for byte.</summary>
    [JsonPropertyName("custom_id")] public required string CustomId { get; init; }
    /// <summary>Always <c>POST</c>.</summary>
    [JsonPropertyName("method")] public string Method { get; init; } = "POST";
    /// <summary>Route identifier <c>/v1/basal</c>; never fetched.</summary>
    [JsonPropertyName("url")] public string Url { get; init; } = "/v1/basal";
    /// <summary>Native /v1/basal request, e.g. <c>JsonSerializer.SerializeToNode(decisionRequest)</c>.
    /// All lines must resolve to the same model; <c>stream: true</c> is refused.</summary>
    [JsonPropertyName("body")] public required JsonNode Body { get; init; }
    /// <summary>Additional fields. The current server refuses unknown fields.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Native answer of one record in a result line.</summary>
public sealed class BatchResponse
{
    /// <summary>200, or the native error status (e.g. 422) with its error body.</summary>
    [JsonPropertyName("status_code")] public int StatusCode { get; init; }
    /// <summary>Record request ID (<c>batch_req_</c> + 32 hex digits).</summary>
    [JsonPropertyName("request_id")] public string? RequestId { get; init; }
    /// <summary>The native /v1/basal answer as the online endpoint writes it. For status 200 it can be
    /// deserialized as <see cref="DecisionResponse"/>.</summary>
    [JsonPropertyName("body")] public JsonNode? Body { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Why a record has no native answer.</summary>
public sealed class BatchRecordError
{
    /// <summary>For example <c>batch_expired</c>, <c>result_too_large</c>, <c>host_budget_exceeded</c>.</summary>
    [JsonPropertyName("code")] public required string Code { get; init; }
    /// <summary>Message.</summary>
    [JsonPropertyName("message")] public string? Message { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>One line of a result file (output or error file). Parse each downloaded line explicitly.</summary>
public sealed class BatchResultLine
{
    /// <summary>Record request ID.</summary>
    [JsonPropertyName("id")] public required string Id { get; init; }
    /// <summary>custom_id of the input line.</summary>
    [JsonPropertyName("custom_id")] public required string CustomId { get; init; }
    /// <summary>Native answer, or null.</summary>
    [JsonPropertyName("response")] public BatchResponse? Response { get; init; }
    /// <summary>Error without a native answer, or null.</summary>
    [JsonPropertyName("error")] public BatchRecordError? Error { get; init; }
    /// <summary>Fields added by a newer server.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

public sealed partial class BasalClient
{
    private const int MaxFilenameBytes = 255;
    private const int MaxIdempotencyKeyLength = 128;
    private const int MaxResourceIdLength = 256;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    /// <summary>POST /v1/files: upload a JSONL input file (purpose <c>batch</c>) from caller-owned bytes.</summary>
    /// <param name="content">File bytes; the SDK does not copy or retain them after the call.</param>
    /// <param name="filename">1–255 UTF-8 bytes without control characters, <c>"</c> or <c>\</c>. Stored as given, never used as a path.</param>
    /// <param name="expiresAfter">File lifetime; omitted uses the server default.</param>
    /// <param name="cancellationToken">Cancels the upload.</param>
    public Task<FileObject> UploadFileAsync(ReadOnlyMemory<byte> content, string filename, ExpiresAfter? expiresAfter = null,
        CancellationToken cancellationToken = default)
        => UploadAsync(new ReadOnlyMemoryContent(content), filename, expiresAfter, cancellationToken);

    /// <summary>POST /v1/files: upload a JSONL input file (purpose <c>batch</c>) streamed from a caller-owned stream.</summary>
    /// <param name="content">Readable stream, read once from its current position to its end. The SDK never disposes
    /// or rewinds it, so the request is not retried.</param>
    /// <param name="filename">1–255 UTF-8 bytes without control characters, <c>"</c> or <c>\</c>. Stored as given, never used as a path.</param>
    /// <param name="expiresAfter">File lifetime; omitted uses the server default.</param>
    /// <param name="cancellationToken">Cancels the upload.</param>
    public Task<FileObject> UploadFileAsync(Stream content, string filename, ExpiresAfter? expiresAfter = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(content);
        if (!content.CanRead) throw new ArgumentException("Expected a readable stream", nameof(content));
        return UploadAsync(new CallerStreamContent(content), filename, expiresAfter, cancellationToken);
    }

    private Task<FileObject> UploadAsync(HttpContent file, string filename, ExpiresAfter? expiresAfter,
        CancellationToken cancellationToken)
    {
        MultipartFormDataContent? form = null;
        try
        {
            var quotedName = Filename(filename);
            form = new MultipartFormDataContent { HeaderEncodingSelector = static (_, _) => StrictUtf8 };
            // The server reserves the upload before the first file byte: expiry fields must precede the file part.
            if (expiresAfter is not null)
            {
                if (expiresAfter.Extra is { Count: > 0 })
                    throw new ArgumentException("Upload expiry has no additional fields", nameof(expiresAfter));
                FormText(form, "expires_after[anchor]", FormValue(expiresAfter.Anchor, nameof(expiresAfter)));
                FormText(form, "expires_after[seconds]", expiresAfter.Seconds.ToString(CultureInfo.InvariantCulture));
            }
            FormText(form, "purpose", "batch");
            file.Headers.TryAddWithoutValidation("Content-Disposition", $"form-data; name=\"file\"; filename=\"{quotedName}\"");
            file.Headers.ContentType = new MediaTypeHeaderValue("application/jsonl");
            form.Add(file);
        }
        catch
        {
            if (form is null) file.Dispose(); else form.Dispose();
            throw;
        }
        var request = new HttpRequestMessage(HttpMethod.Post, new Uri(baseUrl, "v1/files")) { Content = form };
        return SendJsonAsync<FileObject>(request, cancellationToken);
    }

    private static void FormText(MultipartFormDataContent form, string name, string value)
    {
        var part = new ByteArrayContent(Encoding.ASCII.GetBytes(value));
        part.Headers.TryAddWithoutValidation("Content-Disposition", $"form-data; name=\"{name}\"");
        form.Add(part);
    }

    private static string FormValue(string? value, string name)
    {
        if (string.IsNullOrEmpty(value) || value.Length > 256 || !value.All(c => c is > ' ' and < '\x7f'))
            throw new ArgumentException("Expected 1–256 visible ASCII characters", name);
        return value;
    }

    private static string Filename(string filename)
    {
        ArgumentNullException.ThrowIfNull(filename);
        int bytes;
        try { bytes = StrictUtf8.GetByteCount(filename); }
        catch (EncoderFallbackException error) { throw new ArgumentException("The filename is not valid Unicode", nameof(filename), error); }
        // Control characters are refused by the server; quotes and backslashes cannot be framed unambiguously.
        if (bytes is 0 or > MaxFilenameBytes || filename.Any(c => char.IsControl(c) || c is '"' or '\\'))
            throw new ArgumentException("Expected 1–255 UTF-8 bytes without control characters, '\"' or '\\'", nameof(filename));
        return filename;
    }

    /// <summary>GET /v1/files, newest first.</summary>
    /// <param name="after">ID of the last item of the previous page.</param>
    /// <param name="limit">Page size; the server accepts 1–100 (default 20).</param>
    /// <param name="purpose"><c>batch</c> or <c>batch_output</c>; omitted lists both.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public Task<ListPage<FileObject>> ListFilesAsync(string? after = null, int? limit = null, string? purpose = null,
        CancellationToken cancellationToken = default)
        => SendAsync<ListPage<FileObject>>(HttpMethod.Get, "v1/files" + Query(after, limit, purpose), null, cancellationToken);

    /// <summary>GET /v1/files/{file_id}.</summary>
    public Task<FileObject> GetFileAsync(string fileId, CancellationToken cancellationToken = default)
        => SendAsync<FileObject>(HttpMethod.Get, "v1/files/" + PathId(fileId, nameof(fileId)), null, cancellationToken);

    /// <summary>DELETE /v1/files/{file_id}. Input files of active batches are refused with 409.</summary>
    public Task<DeletedFile> DeleteFileAsync(string fileId, CancellationToken cancellationToken = default)
        => SendAsync<DeletedFile>(HttpMethod.Delete, "v1/files/" + PathId(fileId, nameof(fileId)), null, cancellationToken);

    /// <summary>GET /v1/files/{file_id}/content into memory. The raw bytes are returned without JSON parsing.</summary>
    /// <param name="fileId">File ID.</param>
    /// <param name="maxBytes">Largest accepted content; defaults to the client's maxResponseBytes. Raise it explicitly
    /// for a large result file, or stream it with the destination overload.</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    public async Task<byte[]> DownloadFileContentAsync(string fileId, int? maxBytes = null,
        CancellationToken cancellationToken = default)
    {
        var limit = maxBytes is int value ? ValidateLimit(value) : maxResponseBytes;
        using var request = ContentRequest(fileId);
        using var buffer = new MemoryStream();
        await ExchangeAsync(request, buffer, limit, cancellationToken, "application/jsonl", "application/json").ConfigureAwait(false);
        return buffer.ToArray();
    }

    /// <summary>GET /v1/files/{file_id}/content streamed into a caller-owned destination, without JSON parsing or
    /// buffering the whole file. The destination is not disposed; after a failure it may hold a partial copy.</summary>
    /// <param name="fileId">File ID.</param>
    /// <param name="destination">Writable stream.</param>
    /// <param name="maxBytes">Largest accepted content in bytes (required, positive).</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    /// <returns>Number of bytes written.</returns>
    public async Task<long> DownloadFileContentAsync(string fileId, Stream destination, long maxBytes,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (!destination.CanWrite) throw new ArgumentException("Expected a writable stream", nameof(destination));
        if (maxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxBytes), "Response byte limit must be positive");
        using var request = ContentRequest(fileId);
        return await ExchangeAsync(request, destination, maxBytes, cancellationToken, "application/jsonl", "application/json")
            .ConfigureAwait(false);
    }

    private HttpRequestMessage ContentRequest(string fileId)
        => new(HttpMethod.Get, new Uri(baseUrl, "v1/files/" + PathId(fileId, nameof(fileId)) + "/content"));

    /// <summary>POST /v1/batches. Without an idempotency key every call creates a new batch.</summary>
    /// <param name="request">Batch definition.</param>
    /// <param name="idempotencyKey">Optional 1–128 visible ASCII characters: a replay with the same data returns the
    /// existing batch, with different data a 409. Use it to repeat a create safely after a lost answer.</param>
    /// <param name="cancellationToken">Cancels the request; the server may still create the batch.</param>
    public Task<Batch> CreateBatchAsync(CreateBatchRequest request, string? idempotencyKey = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        Questions.Text(request.InputFileId, nameof(request.InputFileId));
        if (idempotencyKey is not null && (idempotencyKey.Length is 0 or > MaxIdempotencyKeyLength
            || !idempotencyKey.All(c => c is > ' ' and < '\x7f')))
            throw new ArgumentException("Expected 1–128 visible ASCII characters", nameof(idempotencyKey));
        var message = new HttpRequestMessage(HttpMethod.Post, new Uri(baseUrl, "v1/batches"))
        {
            Content = new StringContent(JsonSerializer.Serialize(request, JsonOptions), Encoding.UTF8, "application/json")
        };
        if (idempotencyKey is not null) message.Headers.Add("Idempotency-Key", idempotencyKey);
        return SendJsonAsync<Batch>(message, cancellationToken);
    }

    /// <summary>GET /v1/batches, newest first.</summary>
    /// <param name="after">ID of the last item of the previous page.</param>
    /// <param name="limit">Page size; the server accepts 1–100 (default 20).</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public Task<ListPage<Batch>> ListBatchesAsync(string? after = null, int? limit = null,
        CancellationToken cancellationToken = default)
        => SendAsync<ListPage<Batch>>(HttpMethod.Get, "v1/batches" + Query(after, limit, null), null, cancellationToken);

    /// <summary>GET /v1/batches/{batch_id}. One request; poll with your own interval and deadline.</summary>
    public Task<Batch> GetBatchAsync(string batchId, CancellationToken cancellationToken = default)
        => SendAsync<Batch>(HttpMethod.Get, "v1/batches/" + PathId(batchId, nameof(batchId)), null, cancellationToken);

    /// <summary>POST /v1/batches/{batch_id}/cancel. Returns the batch, usually <c>cancelling</c>.</summary>
    public Task<Batch> CancelBatchAsync(string batchId, CancellationToken cancellationToken = default)
        => SendAsync<Batch>(HttpMethod.Post, "v1/batches/" + PathId(batchId, nameof(batchId)) + "/cancel",
            new JsonObject(), cancellationToken);

    /// <summary>One escaped path segment. Dot segments would be normalized away by Uri and are refused.</summary>
    private static string PathId(string id, string name)
    {
        Questions.Text(id, name);
        if (id.Length > MaxResourceIdLength || id is "." or "..")
            throw new ArgumentException("Invalid resource ID", name);
        return Uri.EscapeDataString(id);
    }

    private static string Query(string? after, int? limit, string? purpose)
    {
        var query = new StringBuilder();
        void Add(string key, string value) => query.Append(query.Length == 0 ? '?' : '&').Append(key).Append('=').Append(Uri.EscapeDataString(value));
        if (after is not null) { Questions.Text(after, nameof(after)); Add("after", after); }
        if (limit is int count)
        {
            if (count <= 0) throw new ArgumentOutOfRangeException(nameof(limit), "Limit must be positive");
            Add("limit", count.ToString(CultureInfo.InvariantCulture));
        }
        if (purpose is not null) { Questions.Text(purpose, nameof(purpose)); Add("purpose", purpose); }
        return query.ToString();
    }

    /// <summary>Upload part reading a caller-owned stream once, without disposing it.</summary>
    private sealed class CallerStreamContent(Stream source) : HttpContent
    {
        protected override Task SerializeToStreamAsync(Stream stream, System.Net.TransportContext? context)
            => source.CopyToAsync(stream);

        protected override Task SerializeToStreamAsync(Stream stream, System.Net.TransportContext? context,
            CancellationToken cancellationToken)
            => source.CopyToAsync(stream, cancellationToken);

        protected override bool TryComputeLength(out long length)
        {
            length = 0;
            return false;
        }
    }
}
