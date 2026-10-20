//! Submit native requests, or pass an existing JSONL path as the first argument for the Files flow.
use std::time::Duration;

use basal_client::{
    BatchInputLine, BatchResultsOptions, BatchWaitOptions, Client, ClientOptions, CreateBatchFromRequests,
    CreateBatchRequest, DecisionRequest, FileUploadOptions, Question, QuestionType,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("BASAL_BATCH_TOKEN")?;
    let base_url = std::env::var("BASAL_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".into());
    let client =
        Client::with_options(ClientOptions { base_url: &base_url, api_key: Some(&token), ..Default::default() })?;

    let batch = if let Some(path) = std::env::args().nth(1) {
        // Explicit file input: upload first, then create using its id.
        let data = tokio::task::spawn_blocking(move || std::fs::read(path)).await??;
        let file = client.files().create("input.jsonl", &data, FileUploadOptions::default()).await?;
        client.batches().create(&CreateBatchRequest::new(file.id), None).await?
    } else {
        // Simplified input: the SDK prepares JSONL, uploads it, and creates the batch.
        let question = Question::new(QuestionType::Noul, "Is this urgent?");
        let mut requests = Vec::new();
        for (id, text) in [("ticket-1", "The service is down."), ("ticket-2", "Please send an invoice.")] {
            let mut body = DecisionRequest::new(text);
            body.insert_question("urgent", &question)?;
            requests.push(BatchInputLine::new(id, body));
        }
        client.batches().create(&CreateBatchFromRequests::new(&requests), None).await?
    };
    // Persist this id before waiting. The program can resume retrieval/results later by id.
    println!("Batch: {}", batch.id);
    let final_state = client
        .batches()
        .wait(&batch.id, BatchWaitOptions { timeout: Duration::from_secs(600), ..Default::default() })
        .await?;
    println!("Status: {}", final_state.status);
    if let Some(errors) = &final_state.errors {
        eprintln!("Batch validation errors: {}", errors.data.len());
    }
    // Each file is buffered, here with an explicit 64 MiB allowance.
    let mut results = client
        .batches()
        .results_with_options(&batch.id, BatchResultsOptions { max_bytes: Some(64 * 1024 * 1024) })
        .await?;
    while let Some(result) = results.next().await? {
        println!("{}", serde_json::to_string(&result)?);
    }
    Ok(())
}
