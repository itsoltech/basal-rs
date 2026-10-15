use basal_client::{choice, noul, Client, Questions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://127.0.0.1:8000")?;
    let department = choice(
        "Do którego działu skierować zgłoszenie?",
        &[("returns", "Zwroty"), ("support", "Wsparcie techniczne")],
    )?;
    let urgent = noul("Czy sprawa jest pilna?")?;
    let questions = Questions::new().with("department", &department)?.with("urgent", &urgent)?;
    let response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", &questions).await?;
    println!("Dział: {}, P(tak): {}", response.answer(&department)?.choice, response.answer(&urgent)?.noul);
    Ok(())
}
