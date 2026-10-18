//! Experimental task example. Requires a server with the Basal 2.1 task layer.
use basal_client::{
    classify, compress, extract, questions, verify, ClassifyOptions, Client, CompressOptions, VerifyOptions,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://127.0.0.1:8000")?;
    let definitions = questions! {
        category: classify("", json!({"invoice":"Invoice", "other":"Other document"}), ClassifyOptions::default())?,
        claims: verify("", json!({"total":"The total is 12.50 PLN."}), VerifyOptions::default())?,
        summary: compress("What is the total?", CompressOptions::default())?,
        invoice: extract("", json!({"type":"object","properties":{"total":{"type":"number"}}}))?,
    }?;
    let response = client.decide("Total: 12.50 PLN", &definitions).await?;
    let answers = response.answers()?;
    println!("{:?}: {} / {:?}", answers.category.label, answers.summary.compressed, answers.invoice.data);
    Ok(())
}
