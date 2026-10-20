using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ITSOL.Basal;

/// <summary>Both batch submission paths. Run with <c>dotnet run -- batch</c> and <c>BASAL_BATCH_TOKEN</c> against a server with the Batch API enabled.</summary>
internal static class BatchFlows
{
    public static async Task RunAsync(BasalClient client, CancellationToken cancellationToken = default)
    {
        var body = JsonNode.Parse("""{"state":"A damaged phone.","questions":{"urgent":{"type":"noul","instructions":"Is it urgent?"}}}""")!;

        // OpenAI-shaped resources: upload a JSONL file, create, retrieve, read the content.
        var jsonl = JsonSerializer.Serialize(new BatchInputLine { CustomId = "ticket-1", Body = body.DeepClone() }) + "\n";
        var input = await client.Files.CreateAsync(Encoding.UTF8.GetBytes(jsonl), "tickets.jsonl", cancellationToken: cancellationToken);
        var batch = await client.Batches.CreateAsync(new CreateBatchRequest { InputFileId = input.Id },
            cancellationToken: cancellationToken);
        batch = await client.Batches.RetrieveAsync(batch.Id, cancellationToken);
        if (batch.OutputFileId is string outputId)
            Console.WriteLine($"{outputId}: {(await client.Files.ContentAsync(outputId, cancellationToken: cancellationToken)).Length} bytes");

        // Simplified records: validate, upload and create in one call, then wait and read results.
        try
        {
            batch = await client.Batches.CreateAsync(
                [new BatchInputLine { CustomId = "ticket-2", Body = body.DeepClone() }],
                new CreateBatchOptions { Metadata = new Dictionary<string, string> { ["job"] = "tickets" } }, cancellationToken);
        }
        catch (Exception error) when (error is IBatchSubmissionFailure failure)
        {
            // The input file exists; resume with the same file and key instead of uploading again.
            Console.WriteLine($"Create failed after uploading {failure.FileId}; resuming.");
            batch = await client.Batches.CreateAsync(failure.Request, failure.IdempotencyKey, cancellationToken);
        }
        try
        {
            batch = await client.Batches.WaitAsync(batch.Id, TimeSpan.FromMinutes(30), cancellationToken: cancellationToken);
        }
        catch (BatchWaitTimeoutException timeout)
        {
            Console.WriteLine($"{timeout.BatchId} still {timeout.LastBatch?.Status}; the server keeps running it.");
            return;
        }
        await foreach (var line in client.Batches.ResultsAsync(batch.Id, cancellationToken: cancellationToken))
            Console.WriteLine($"{line.CustomId}: {line.Response?.StatusCode.ToString() ?? line.Error?.Code}");
    }
}
