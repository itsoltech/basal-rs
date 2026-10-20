using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Unicode;

namespace ITSOL.Basal;

/// <summary>Options of <see cref="BasalBatches.CreateAsync(IEnumerable{BatchInputLine}, CreateBatchOptions?, CancellationToken)"/>:
/// the batch parameters of <see cref="CreateBatchRequest"/> plus the generated input file.</summary>
public sealed class CreateBatchOptions
{
    /// <summary>Endpoint of every record; the server accepts only <c>/v1/basal</c>.</summary>
    public string Endpoint { get; init; } = "/v1/basal";
    /// <summary>The server accepts only <c>24h</c>.</summary>
    public string CompletionWindow { get; init; } = "24h";
    /// <summary>Up to 16 string pairs, checked by the server.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }
    /// <summary>Lifetime of each result file after its creation; omitted uses the server default.</summary>
    public ExpiresAfter? OutputExpiresAfter { get; init; }
    /// <summary>1–128 visible ASCII characters sent with the create request. When omitted, a fresh key is generated
    /// for this call and reported by <see cref="IBatchSubmissionFailure.IdempotencyKey"/> on failure.</summary>
    public string? IdempotencyKey { get; init; }
    /// <summary>Name of the uploaded input file (same rules as <see cref="BasalClient.UploadFileAsync(ReadOnlyMemory{byte}, string, ExpiresAfter?, CancellationToken)"/>).</summary>
    public string Filename { get; init; } = "batch.jsonl";
    /// <summary>Lifetime of the uploaded input file; omitted uses the server default.</summary>
    public ExpiresAfter? InputExpiresAfter { get; init; }
}

/// <summary>The input file was uploaded but the batch was not confirmed as created. Resume with
/// <c>client.Batches.CreateAsync(failure.Request, failure.IdempotencyKey)</c> instead of uploading again: the same key
/// and data return the batch if the lost create did succeed. The SDK neither retries nor deletes the file.</summary>
public interface IBatchSubmissionFailure
{
    /// <summary>The uploaded input file.</summary>
    FileObject InputFile { get; }
    /// <summary>ID of the uploaded input file.</summary>
    string FileId { get; }
    /// <summary>Idempotency key sent with the create request, generated when the caller supplied none.</summary>
    string IdempotencyKey { get; }
    /// <summary>The create request that failed, referencing <see cref="FileId"/>.</summary>
    CreateBatchRequest Request { get; }
}

/// <summary>Creating the batch failed after its input file was uploaded. The original HTTP, transport or response
/// error is in <see cref="Exception.InnerException"/>.</summary>
public sealed class BatchSubmissionException : BasalException, IBatchSubmissionFailure
{
    internal BatchSubmissionException(FileObject inputFile, string idempotencyKey, CreateBatchRequest request, Exception cause)
        : base($"Creating the batch failed after uploading input file {inputFile.Id}", cause)
        => (InputFile, IdempotencyKey, Request) = (inputFile, idempotencyKey, request);

    /// <inheritdoc/>
    public FileObject InputFile { get; }
    /// <inheritdoc/>
    public string FileId => InputFile.Id;
    /// <inheritdoc/>
    public string IdempotencyKey { get; }
    /// <inheritdoc/>
    public CreateBatchRequest Request { get; }
}

/// <summary>The caller cancelled after the input file was uploaded. It remains an
/// <see cref="OperationCanceledException"/> for the caller's token; the server may still have created the batch.</summary>
public sealed class BatchSubmissionCanceledException : OperationCanceledException, IBatchSubmissionFailure
{
    internal BatchSubmissionCanceledException(FileObject inputFile, string idempotencyKey, CreateBatchRequest request,
        OperationCanceledException cause, CancellationToken token)
        : base($"Batch creation was cancelled after uploading input file {inputFile.Id}", cause, token)
        => (InputFile, IdempotencyKey, Request) = (inputFile, idempotencyKey, request);

    /// <inheritdoc/>
    public FileObject InputFile { get; }
    /// <inheritdoc/>
    public string FileId => InputFile.Id;
    /// <inheritdoc/>
    public string IdempotencyKey { get; }
    /// <inheritdoc/>
    public CreateBatchRequest Request { get; }
}

/// <summary><see cref="BasalBatches.WaitAsync"/> reached its timeout. The batch is not cancelled on the server.</summary>
public sealed class BatchWaitTimeoutException : BasalException
{
    internal BatchWaitTimeoutException(string batchId, Batch? lastBatch, TimeSpan timeout, Exception cause)
        : base($"Batch {batchId} did not reach a terminal status within {timeout}"
            + (lastBatch is null ? "" : $"; last status {lastBatch.Status}"), cause)
        => (BatchId, LastBatch, Timeout) = (batchId, lastBatch, timeout);

    /// <summary>Waited batch ID.</summary>
    public string BatchId { get; }
    /// <summary>Last retrieved batch, or null when no retrieval completed.</summary>
    public Batch? LastBatch { get; }
    /// <summary>Effective timeout.</summary>
    public TimeSpan Timeout { get; }
}

/// <summary><see cref="BasalBatches.ResultsAsync"/> found a batch that is not terminal. Results are not waited for.</summary>
public sealed class BatchNotReadyException : BasalException
{
    internal BatchNotReadyException(Batch batch)
        : base($"Batch {batch.Id} is {batch.Status}, not terminal; wait for it before reading results")
        => Batch = batch;

    /// <summary>The retrieved batch.</summary>
    public Batch Batch { get; }
}

/// <summary>A result file line is not UTF-8, not JSON or not a result envelope. The line content is not included.</summary>
public sealed class BatchResultFormatException : BasalException
{
    internal BatchResultFormatException(string fileId, long line, string problem, Exception? cause = null)
        : base($"Invalid result line {line} in file {fileId}: {problem}", cause)
        => (FileId, Line) = (fileId, line);

    /// <summary>Result file ID.</summary>
    public string FileId { get; }
    /// <summary>1-based line number.</summary>
    public long Line { get; }
}

/// <summary><c>client.Files</c>: the Files API under OpenAI-style names. Each method is the flat method named in its
/// summary, with the same validation, limits, errors and cancellation.</summary>
public sealed class BasalFiles
{
    private readonly BasalClient client;

    internal BasalFiles(BasalClient client) => this.client = client;

    /// <summary>Upload caller-owned bytes as a <c>batch</c> input file (<see cref="BasalClient.UploadFileAsync(ReadOnlyMemory{byte}, string, ExpiresAfter?, CancellationToken)"/>).</summary>
    /// <param name="content">File bytes; not copied or retained after the call.</param>
    /// <param name="filename">1–255 UTF-8 bytes without control characters, <c>"</c> or <c>\</c>.</param>
    /// <param name="expiresAfter">File lifetime; omitted uses the server default.</param>
    /// <param name="cancellationToken">Cancels the upload.</param>
    public Task<FileObject> CreateAsync(ReadOnlyMemory<byte> content, string filename, ExpiresAfter? expiresAfter = null,
        CancellationToken cancellationToken = default)
        => client.UploadFileAsync(content, filename, expiresAfter, cancellationToken);

    /// <summary>Upload a caller-owned stream, read once and never disposed or rewound
    /// (<see cref="BasalClient.UploadFileAsync(Stream, string, ExpiresAfter?, CancellationToken)"/>).</summary>
    /// <param name="content">Readable stream.</param>
    /// <param name="filename">1–255 UTF-8 bytes without control characters, <c>"</c> or <c>\</c>.</param>
    /// <param name="expiresAfter">File lifetime; omitted uses the server default.</param>
    /// <param name="cancellationToken">Cancels the upload.</param>
    public Task<FileObject> CreateAsync(Stream content, string filename, ExpiresAfter? expiresAfter = null,
        CancellationToken cancellationToken = default)
        => client.UploadFileAsync(content, filename, expiresAfter, cancellationToken);

    /// <summary>File metadata (<see cref="BasalClient.GetFileAsync"/>).</summary>
    public Task<FileObject> RetrieveAsync(string fileId, CancellationToken cancellationToken = default)
        => client.GetFileAsync(fileId, cancellationToken);

    /// <summary>One page of files, newest first (<see cref="BasalClient.ListFilesAsync"/>).</summary>
    /// <param name="after">ID of the last item of the previous page.</param>
    /// <param name="limit">Page size; the server accepts 1–100.</param>
    /// <param name="purpose"><c>batch</c> or <c>batch_output</c>; omitted lists both.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public Task<ListPage<FileObject>> ListAsync(string? after = null, int? limit = null, string? purpose = null,
        CancellationToken cancellationToken = default)
        => client.ListFilesAsync(after, limit, purpose, cancellationToken);

    /// <summary>Delete a file (<see cref="BasalClient.DeleteFileAsync"/>).</summary>
    public Task<DeletedFile> DeleteAsync(string fileId, CancellationToken cancellationToken = default)
        => client.DeleteFileAsync(fileId, cancellationToken);

    /// <summary>Raw file content in memory, within <paramref name="maxBytes"/> or the client's maxResponseBytes
    /// (<see cref="BasalClient.DownloadFileContentAsync(string, int?, CancellationToken)"/>).</summary>
    /// <param name="fileId">File ID.</param>
    /// <param name="maxBytes">Largest accepted content.</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    public Task<byte[]> ContentAsync(string fileId, int? maxBytes = null, CancellationToken cancellationToken = default)
        => client.DownloadFileContentAsync(fileId, maxBytes, cancellationToken);

    /// <summary>Raw file content copied into a caller-owned stream, not disposed
    /// (<see cref="BasalClient.DownloadFileContentAsync(string, Stream, long, CancellationToken)"/>).</summary>
    /// <param name="fileId">File ID.</param>
    /// <param name="destination">Writable stream; may hold a partial copy after a failure.</param>
    /// <param name="maxBytes">Largest accepted content in bytes.</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    /// <returns>Number of bytes written.</returns>
    public Task<long> ContentAsync(string fileId, Stream destination, long maxBytes, CancellationToken cancellationToken = default)
        => client.DownloadFileContentAsync(fileId, destination, maxBytes, cancellationToken);
}

/// <summary><c>client.Batches</c>: the Batch API under OpenAI-style names, plus submission, wait and result helpers.
/// No method retries, deletes files or cancels batches on its own.</summary>
public sealed class BasalBatches
{
    /// <summary>Largest number of records accepted by <see cref="CreateAsync(IEnumerable{BatchInputLine}, CreateBatchOptions?, CancellationToken)"/> (the server default).</summary>
    public const int MaxRequests = 50_000;
    /// <summary>Largest generated JSONL input, including line feeds (the server default).</summary>
    public const int MaxInputBytes = 200_000_000;
    private const int MaxCustomIdBytes = 256;
    private static readonly TimeSpan DefaultWaitTimeout = TimeSpan.FromSeconds(600);
    private static readonly TimeSpan DefaultPollInterval = TimeSpan.FromSeconds(5);
    private readonly BasalClient client;

    internal BasalBatches(BasalClient client) => this.client = client;

    /// <summary>Create a batch from an uploaded input file (<see cref="BasalClient.CreateBatchAsync"/>).</summary>
    /// <param name="request">Batch definition referencing <see cref="CreateBatchRequest.InputFileId"/>.</param>
    /// <param name="idempotencyKey">Optional 1–128 visible ASCII characters; without it every call creates a new batch.</param>
    /// <param name="cancellationToken">Cancels the request; the server may still create the batch.</param>
    public Task<Batch> CreateAsync(CreateBatchRequest request, string? idempotencyKey = null,
        CancellationToken cancellationToken = default)
        => client.CreateBatchAsync(request, idempotencyKey, cancellationToken);

    /// <summary>Validate the records, serialize them as one JSONL input in memory, upload it (purpose <c>batch</c>)
    /// and create the batch.</summary>
    /// <remarks>Before any HTTP the records are checked: at least one and at most <see cref="MaxRequests"/>; no null
    /// record; <c>custom_id</c> of 1–256 UTF-8 bytes and unique byte for byte; method <c>POST</c>, url
    /// <c>/v1/basal</c>, a JSON object body and no extra fields; at most <see cref="MaxInputBytes"/> of JSONL; valid
    /// options and idempotency key. The whole JSONL is buffered in memory in one growing buffer (while it grows,
    /// transiently up to about three times the JSONL size, plus one serialized record) and sent with the client
    /// timeout covering the whole upload. The enumerable is read once.
    /// The idempotency key protects only the create request; when omitted, one is generated for this call.
    /// An upload failure propagates unchanged. A later failure raises <see cref="BatchSubmissionException"/>, or
    /// <see cref="BatchSubmissionCanceledException"/> on caller cancellation; both implement
    /// <see cref="IBatchSubmissionFailure"/> with the file ID and key. The file is not deleted.</remarks>
    /// <param name="requests">Records; <see cref="BatchInputLine.Body"/> is a native /v1/basal request object.</param>
    /// <param name="options">Batch parameters, idempotency key and input file name/expiry.</param>
    /// <param name="cancellationToken">Cancels the upload or create.</param>
    public async Task<Batch> CreateAsync(IEnumerable<BatchInputLine> requests, CreateBatchOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        options ??= new CreateBatchOptions();
        if (options.Endpoint != "/v1/basal" || options.CompletionWindow != "24h")
            throw new ArgumentException("Batch requests require endpoint /v1/basal and completion window 24h", nameof(options));
        BasalClient.ValidateFilename(options.Filename);
        var key = options.IdempotencyKey ?? "basal-sdk-" + Guid.NewGuid().ToString("N");
        BasalClient.ValidateIdempotencyKey(key, nameof(options));
        using var jsonl = Serialize(requests);
        var input = await client.UploadFileAsync(jsonl.GetBuffer().AsMemory(0, (int)jsonl.Length), options.Filename,
            options.InputExpiresAfter, cancellationToken).ConfigureAwait(false);
        var request = new CreateBatchRequest
        {
            InputFileId = input.Id,
            Endpoint = options.Endpoint,
            CompletionWindow = options.CompletionWindow,
            Metadata = options.Metadata,
            OutputExpiresAfter = options.OutputExpiresAfter,
        };
        try
        {
            return await client.CreateBatchAsync(request, key, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException error)
        {
            var token = cancellationToken.IsCancellationRequested ? cancellationToken : error.CancellationToken;
            throw new BatchSubmissionCanceledException(input, key, request, error, token);
        }
        catch (Exception error)
        {
            throw new BatchSubmissionException(input, key, request, error);
        }
    }

    /// <summary>Serialize validated records to LF-terminated JSONL within <see cref="MaxInputBytes"/>.</summary>
    private static MemoryStream Serialize(IEnumerable<BatchInputLine> requests)
    {
        var buffer = new MemoryStream();
        try
        {
            using var writer = new Utf8JsonWriter(buffer);
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var line in requests)
            {
                var index = seen.Count;
                if (index == MaxRequests)
                    throw new ArgumentException($"A batch accepts at most {MaxRequests} records", nameof(requests));
                if (line is null) throw new ArgumentException($"Record {index} is null", nameof(requests));
                ValidateRecord(line, index);
                if (!seen.Add(line.CustomId))
                    throw new ArgumentException($"Record {index} repeats a custom_id", nameof(requests));
                writer.Reset(buffer);
                try
                {
                    writer.WriteStartObject();
                    writer.WriteString("custom_id", line.CustomId);
                    writer.WriteString("method", line.Method);
                    writer.WriteString("url", line.Url);
                    writer.WritePropertyName("body");
                    line.Body.WriteTo(writer);
                    writer.WriteEndObject();
                    writer.Flush();
                }
                catch (Exception error) when (error is JsonException or NotSupportedException or InvalidOperationException)
                {
                    throw new ArgumentException($"Record {index} body cannot be serialized", nameof(requests), error);
                }
                if (buffer.Length + 1 > MaxInputBytes)
                    throw new ArgumentException($"The JSONL input exceeds {MaxInputBytes} bytes", nameof(requests));
                buffer.WriteByte((byte)'\n');
            }
            if (seen.Count == 0) throw new ArgumentException("A batch requires at least one record", nameof(requests));
            return buffer;
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }

    private static void ValidateRecord(BatchInputLine line, int index)
    {
        const string name = "requests";
        if (line.CustomId is null) throw new ArgumentException($"Record {index} has no custom_id", name);
        int bytes;
        try { bytes = BasalClient.StrictUtf8Encoding.GetByteCount(line.CustomId); }
        catch (EncoderFallbackException error)
        {
            throw new ArgumentException($"Record {index} custom_id is not valid Unicode", name, error);
        }
        if (bytes is 0 or > MaxCustomIdBytes)
            throw new ArgumentException($"Record {index} custom_id must have 1–{MaxCustomIdBytes} UTF-8 bytes", name);
        if (line.Method != "POST") throw new ArgumentException($"Record {index} method must be POST", name);
        if (line.Url != "/v1/basal") throw new ArgumentException($"Record {index} url must be /v1/basal", name);
        if (line.Body is not JsonObject) throw new ArgumentException($"Record {index} body must be a JSON object", name);
        if (line.Body["stream"] is JsonValue stream && stream.TryGetValue<bool>(out var streaming) && streaming)
            throw new ArgumentException($"Record {index} cannot enable streaming", name);
        if (line.Extra is { Count: > 0 }) throw new ArgumentException($"Record {index} has unsupported fields", name);
    }

    /// <summary>One batch (<see cref="BasalClient.GetBatchAsync"/>).</summary>
    public Task<Batch> RetrieveAsync(string batchId, CancellationToken cancellationToken = default)
        => client.GetBatchAsync(batchId, cancellationToken);

    /// <summary>One page of batches, newest first (<see cref="BasalClient.ListBatchesAsync"/>).</summary>
    /// <param name="after">ID of the last item of the previous page.</param>
    /// <param name="limit">Page size; the server accepts 1–100.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public Task<ListPage<Batch>> ListAsync(string? after = null, int? limit = null, CancellationToken cancellationToken = default)
        => client.ListBatchesAsync(after, limit, cancellationToken);

    /// <summary>Request cancellation (<see cref="BasalClient.CancelBatchAsync"/>); usually returns <c>cancelling</c>.</summary>
    public Task<Batch> CancelAsync(string batchId, CancellationToken cancellationToken = default)
        => client.CancelBatchAsync(batchId, cancellationToken);

    /// <summary>Poll a batch until it is terminal (completed, failed, expired or cancelled) and return it.</summary>
    /// <remarks>The first retrieval is immediate; later ones follow <paramref name="pollInterval"/>. One monotonic
    /// deadline bounds the whole wait, including in-flight HTTP requests and pauses; unknown statuses keep polling until
    /// it. On timeout <see cref="BatchWaitTimeoutException"/> carries the batch ID and last retrieved batch; the server
    /// batch is not cancelled. HTTP and transport errors propagate without retry. A <c>failed</c> batch is returned
    /// with its <see cref="Batch.Errors"/>. Caller cancellation raises <see cref="OperationCanceledException"/>.</remarks>
    /// <param name="batchId">Batch ID.</param>
    /// <param name="timeout">Positive total wait, at most 4294967294 ms; default 600 seconds.</param>
    /// <param name="pollInterval">Positive pause between retrievals, at most 4294967294 ms; default 5 seconds.</param>
    /// <param name="cancellationToken">Cancels the wait.</param>
    public async Task<Batch> WaitAsync(string batchId, TimeSpan? timeout = null, TimeSpan? pollInterval = null,
        CancellationToken cancellationToken = default)
    {
        BasalClient.ValidatePathId(batchId, nameof(batchId));
        var limit = Duration(timeout ?? DefaultWaitTimeout, nameof(timeout));
        var pause = Duration(pollInterval ?? DefaultPollInterval, nameof(pollInterval));
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(limit);
        Batch? last = null;
        try
        {
            for (;;)
            {
                last = await client.GetBatchAsync(batchId, deadline.Token).ConfigureAwait(false);
                if (BatchStatus.IsTerminal(last.Status)) return last;
                await Task.Delay(pause, deadline.Token).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException error) when (!cancellationToken.IsCancellationRequested && deadline.IsCancellationRequested)
        {
            throw new BatchWaitTimeoutException(batchId, last, limit, error);
        }
    }

    private static TimeSpan Duration(TimeSpan value, string name)
        => value > TimeSpan.Zero && value.TotalMilliseconds <= uint.MaxValue - 1
            ? value
            : throw new ArgumentOutOfRangeException(name, "Expected a positive duration of at most 4294967294 ms");

    /// <summary>Result lines of a terminal batch: the output file, then the error file (once if both IDs are equal).</summary>
    /// <remarks>Enumeration first retrieves the batch; a non-terminal batch raises
    /// <see cref="BatchNotReadyException"/> without waiting. Each file is downloaded whole into memory within
    /// <paramref name="maxBytesPerFile"/> (default: the client's maxResponseBytes, 8 MiB unless configured) and then
    /// parsed one line at a time; this is not unbounded streaming. Lines are yielded as written, including record
    /// errors; completed batches can contain failed records, and expired or cancelled batches partial results.
    /// A batch without result files yields nothing; batch-level validation errors are in
    /// <see cref="Batch.Errors"/> from <see cref="RetrieveAsync"/> or <see cref="WaitAsync"/>. Invalid UTF-8,
    /// JSON or envelopes raise <see cref="BatchResultFormatException"/> with the file ID and line number.
    /// Each enumeration repeats the requests.</remarks>
    /// <param name="batchId">Batch ID.</param>
    /// <param name="maxBytesPerFile">Largest accepted size of each result file.</param>
    /// <param name="cancellationToken">Cancels retrieval, downloads and enumeration.</param>
    public IAsyncEnumerable<BatchResultLine> ResultsAsync(string batchId, int? maxBytesPerFile = null,
        CancellationToken cancellationToken = default)
    {
        BasalClient.ValidatePathId(batchId, nameof(batchId));
        if (maxBytesPerFile <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxBytesPerFile), "Response byte limit must be positive");
        return Results(batchId, maxBytesPerFile, cancellationToken);
    }

    private async IAsyncEnumerable<BatchResultLine> Results(string batchId, int? maxBytesPerFile,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var batch = await client.GetBatchAsync(batchId, cancellationToken).ConfigureAwait(false);
        if (!BatchStatus.IsTerminal(batch.Status)) throw new BatchNotReadyException(batch);
        string?[] files = batch.ErrorFileId == batch.OutputFileId ? [batch.OutputFileId] : [batch.OutputFileId, batch.ErrorFileId];
        foreach (var fileId in files)
        {
            if (fileId is null) continue;
            var content = await client.DownloadFileContentAsync(fileId, maxBytesPerFile, cancellationToken).ConfigureAwait(false);
            long number = 0;
            for (var start = 0; start < content.Length;)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var end = Array.IndexOf(content, (byte)'\n', start);
                if (end < 0) end = content.Length;
                var line = new ReadOnlyMemory<byte>(content, start, end - start);
                start = end + 1;
                number++;
                yield return ParseLine(fileId, number, line);
            }
        }
    }

    /// <summary>One validated result line. Empty interior lines fail; CRLF is accepted.</summary>
    private static BatchResultLine ParseLine(string fileId, long number, ReadOnlyMemory<byte> line)
    {
        var bytes = line.Span;
        if (bytes is [.., (byte)'\r']) bytes = bytes[..^1];
        if (bytes.IsEmpty) throw new BatchResultFormatException(fileId, number, "empty line");
        if (!Utf8.IsValid(bytes)) throw new BatchResultFormatException(fileId, number, "not UTF-8");
        BatchResultLine? result;
        try
        {
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("response", out _) || !root.TryGetProperty("error", out _))
                throw new BatchResultFormatException(fileId, number, "missing response or error fields");
            result = root.Deserialize<BatchResultLine>();
        }
        catch (JsonException error) { throw new BatchResultFormatException(fileId, number, "not a JSON result line", error); }
        if (result is null || string.IsNullOrEmpty(result.Id) || string.IsNullOrEmpty(result.CustomId))
            throw new BatchResultFormatException(fileId, number, "missing id or custom_id");
        if ((result.Response is null) == (result.Error is null))
            throw new BatchResultFormatException(fileId, number, "expected exactly one response or error");
        if (result.Response is { } response && (response.StatusCode is < 100 or > 599
            || string.IsNullOrEmpty(response.RequestId) || response.Body is not JsonObject))
            throw new BatchResultFormatException(fileId, number, "invalid response envelope");
        if (result.Error is { } recordError && (string.IsNullOrEmpty(recordError.Code) || recordError.Message is null))
            throw new BatchResultFormatException(fileId, number, "invalid error envelope");
        return result;
    }
}

public sealed partial class BasalClient
{
    /// <summary>Files API resource: <c>CreateAsync</c>, <c>RetrieveAsync</c>, <c>ListAsync</c>, <c>DeleteAsync</c>,
    /// <c>ContentAsync</c>. The flat methods remain available.</summary>
    public BasalFiles Files { get; }

    /// <summary>Batch API resource: <c>CreateAsync</c> from a file or records, <c>RetrieveAsync</c>, <c>ListAsync</c>,
    /// <c>CancelAsync</c>, <c>WaitAsync</c> and <c>ResultsAsync</c>. The flat methods remain available.</summary>
    public BasalBatches Batches { get; }

    internal static UTF8Encoding StrictUtf8Encoding => StrictUtf8;

    internal static void ValidateFilename(string filename) => Filename(filename);

    internal static void ValidateIdempotencyKey(string key, string name) => IdempotencyKey(key, name);

    internal static void ValidatePathId(string id, string name) => PathId(id, name);
}
