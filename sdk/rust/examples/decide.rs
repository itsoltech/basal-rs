use basal_client::{choice, noul, questions, Client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://127.0.0.1:8000")?;
    let department = choice(
        "Do którego działu skierować zgłoszenie?",
        &[("returns", "Zwroty"), ("support", "Wsparcie techniczne")],
    )?;
    let urgent = noul("Czy sprawa jest pilna?")?;
    let questions = questions! {
        department: &department,
        second_department: &department,
        urgent: &urgent,
    }?;
    let response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", &questions).await?;
    let answers = response.answers()?;
    println!("Dział: {}, P(tak): {}", answers.department.choice, answers.urgent.noul);
    println!("Drugie pytanie: {}", answers.second_department.choice);
    Ok(())
}
