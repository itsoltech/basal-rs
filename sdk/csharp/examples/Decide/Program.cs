using ITSOL.Basal;

// Optional: a key from `basal keys create`, needed only when the server has auth.enabled.
var apiKey = Environment.GetEnvironmentVariable("BASAL_API_KEY");
if (args is ["batch", ..])
{
    using var batchClient = new BasalClient(apiKey: apiKey);
    await BatchFlows.RunAsync(batchClient);
    return;
}

using var client = new BasalClient(apiKey: apiKey);
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
