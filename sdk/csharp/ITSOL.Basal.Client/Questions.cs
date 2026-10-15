using System.Text.Json.Nodes;
using System.Text.Json;

namespace ITSOL.Basal;

/// <summary>Display of named criteria keys.</summary>
public enum OptionKeys
{
    /// <summary>Include criterion names in prompts.</summary>
    Show,
    /// <summary>Hide criterion names.</summary>
    Hide
}
/// <summary>Facts preprocessing.</summary>
public enum FactsMode
{
    /// <summary>Do not preprocess facts.</summary>
    Off,
    /// <summary>Let the model preprocess facts.</summary>
    Auto
}
/// <summary>Typed decision options; cancellation is a method argument.</summary>
public sealed record DecisionOptions(string? Model = null, FactsMode? Facts = null);
/// <summary>Choice display/evidence options.</summary>
public sealed record ChoiceOptions(bool Evidence = false, OptionKeys OptionKeys = OptionKeys.Show);
/// <summary>Evidence options for noul/score.</summary>
public sealed record EvidenceOptions(bool Evidence = false);
/// <summary>Multi options deliberately do not include evidence.</summary>
public sealed record MultiOptions(double? Threshold = null, int? Min = null, int? Max = null);
/// <summary>Act options; omitted criteria use true/false outcomes.</summary>
public sealed record ActOptions(IReadOnlyDictionary<string, string>? Criteria = null, bool Evidence = false,
    string? DeferAction = null, double? MaxError = null);

/// <summary>Common type for heterogeneous typed question maps. The wire snapshot is protected.</summary>
public abstract class QuestionSpec
{
    private readonly JsonObject wire;
    /// <summary>Snapshot a validated question.</summary>
    protected QuestionSpec(string type, string instructions, JsonObject fields)
    {
        Questions.Text(instructions, nameof(instructions));
        wire = (JsonObject)fields.DeepClone();
        wire["type"] = type; wire["instructions"] = instructions;
    }
    /// <summary>A detached native JSON copy.</summary>
    public JsonObject ToJson() => (JsonObject)wire.DeepClone();
    /// <summary>Validate an answer when the ergonomic client receives it.</summary>
    public abstract AnswerResult Parse(DecisionResponse response, string id);
    /// <summary>Find an answer with the expected native type.</summary>
    protected Answer Find(DecisionResponse response, string id, string type)
    {
        if (response.Answers is null || !response.Answers.TryGetValue(id, out var answer) || answer is null || answer.Type != type)
            throw new BasalException("Basal returned a missing or mismatched answer");
        if (answer.Probabilities is null || answer.Probabilities.Values.Any(v => !double.IsFinite(v) || v < 0 || v > 1))
            throw new BasalException("Basal returned invalid probabilities");
        if (answer.Evidence is not null && answer.Evidence.Any(span => span is null || span.Text is null || span.Start < 0 || span.End < 0 || !double.IsFinite(span.Probability) || span.Probability < 0 || span.Probability > 1))
            throw new BasalException("Invalid evidence span");
        return answer;
    }
    /// <summary>Required finite numeric result.</summary>
    protected static double Number(double? value) => value is double number && double.IsFinite(number)
        ? number : throw new BasalException("Basal returned a missing or invalid numeric answer field");
}
/// <summary>Typed access to the result of a question.</summary>
public abstract class QuestionSpec<TAnswer> : QuestionSpec where TAnswer : AnswerResult
{
    /// <summary>Initialize a validated question.</summary>
    protected QuestionSpec(string type, string instructions, JsonObject fields) : base(type, instructions, fields) { }
    /// <summary>Read a checked answer with non-null required fields.</summary>
    public abstract TAnswer Read(DecisionResponse response, string id);
    /// <inheritdoc/>
    public sealed override AnswerResult Parse(DecisionResponse response, string id) => Read(response, id);
}
/// <summary>Typed result metadata. Missing evidence/extensions use empty collections.</summary>
public abstract record AnswerResult
{
    /// <summary>Supporting spans, empty when evidence was not returned.</summary>
    public IReadOnlyList<EvidenceSpan> Evidence { get; init; } = Array.Empty<EvidenceSpan>();
    /// <summary>Additional native fields.</summary>
    public IReadOnlyDictionary<string, JsonElement> Extra { get; init; } = new Dictionary<string, JsonElement>();
}
/// <summary>A checked choice result.</summary>
public sealed record ChoiceResult(string Choice, double Confidence, IReadOnlyDictionary<string, double> Probabilities) : AnswerResult;
/// <summary>A checked noul result.</summary>
public sealed record NoulResult(double Noul, double Confidence, IReadOnlyDictionary<string, double> Probabilities) : AnswerResult;
/// <summary>A checked score result.</summary>
public sealed record ScoreResult(double Score, double Confidence, IReadOnlyDictionary<string, string> Legend, IReadOnlyDictionary<string, double> Probabilities) : AnswerResult;
/// <summary>A checked multi result.</summary>
public sealed record MultiResult(IReadOnlyList<string> Selected, double Threshold, double SetConfidence, IReadOnlyDictionary<string, double> Probabilities) : AnswerResult;
/// <summary>A checked act result.</summary>
public sealed record ActResult(string Action, string Outcome, double Confidence, string Calibration, IReadOnlyDictionary<string, double> ExpectedCosts, IReadOnlyDictionary<string, double> Probabilities) : AnswerResult
{
    /// <summary>Native refusal explanation, absent when none was returned.</summary>
    public string? Refused { get; init; }
    /// <summary>Native certified error target warning, absent when none was returned.</summary>
    public string? MaxError { get; init; }
}

/// <summary>Choose one named option.</summary>
public sealed class ChoiceQuestion : QuestionSpec<ChoiceResult>
{
    /// <summary>Create a choice with 2–10 named criteria.</summary>
    public ChoiceQuestion(string instructions, IReadOnlyDictionary<string, string> criteria, ChoiceOptions? options = null)
        : base("choice", instructions, Build(criteria, options ?? new())) { }
    private static JsonObject Build(IReadOnlyDictionary<string, string> criteria, ChoiceOptions options)
    {
        if (!Enum.IsDefined(options.OptionKeys)) throw new ArgumentException("Invalid option keys");
        var fields = Questions.Display(options.Evidence); fields["criteria"] = Questions.Criteria(criteria, 2, 10);
        if (options.OptionKeys == OptionKeys.Hide) fields["option_keys"] = "hide";
        return fields;
    }
    /// <inheritdoc/>
    public override ChoiceResult Read(DecisionResponse response, string id)
    {
        var answer = Find(response, id, "choice");
        if (answer.Choice is null || ToJson()["criteria"]!.AsObject().ContainsKey(answer.Choice) == false)
            throw new BasalException("Basal returned an unknown choice");
        return new(answer.Choice, Number(answer.Confidence), answer.Probabilities) { Evidence = answer.Evidence ?? [], Extra = answer.Extra ?? [] };
    }
}
/// <summary>Read a yes/no probability.</summary>
public sealed class NoulQuestion : QuestionSpec<NoulResult>
{
    /// <summary>Create a yes/no question.</summary>
    public NoulQuestion(string instructions, EvidenceOptions? options = null) : base("noul", instructions, Questions.Display(options?.Evidence ?? false)) { }
    /// <inheritdoc/>
    public override NoulResult Read(DecisionResponse response, string id)
    { var a = Find(response, id, "noul"); return new(Number(a.Noul), Number(a.Confidence), a.Probabilities) { Evidence = a.Evidence ?? [], Extra = a.Extra ?? [] }; }
}
/// <summary>Score over ordered levels.</summary>
public sealed class ScoreQuestion : QuestionSpec<ScoreResult>
{
    /// <summary>Create a score with 2–10 ordered levels.</summary>
    public ScoreQuestion(string instructions, IReadOnlyList<string> levels, EvidenceOptions? options = null)
        : base("score", instructions, Build(levels, options?.Evidence ?? false)) { }
    private static JsonObject Build(IReadOnlyList<string> levels, bool evidence)
    {
        ArgumentNullException.ThrowIfNull(levels);
        if (levels.Count is < 2 or > 10) throw new ArgumentException("Score requires 2–10 levels");
        var fields = Questions.Display(evidence); var array = new JsonArray();
        foreach (var level in levels) { Questions.Text(level, "level"); array.Add(level); }
        fields["criteria"] = array; return fields;
    }
    /// <inheritdoc/>
    public override ScoreResult Read(DecisionResponse response, string id)
    { var a = Find(response, id, "score");
        if (a.Legend is null || a.Legend.Values.Any(value => value is null)) throw new BasalException("Invalid score legend");
        return new(Number(a.Score), Number(a.Confidence), a.Legend ?? throw new BasalException("Missing score legend"), a.Probabilities) { Evidence = a.Evidence ?? [], Extra = a.Extra ?? [] }; }
}
/// <summary>Select labels independently; evidence is unavailable.</summary>
public sealed class MultiQuestion : QuestionSpec<MultiResult>
{
    /// <summary>Create a multi question with one or more labels.</summary>
    public MultiQuestion(string instructions, IReadOnlyDictionary<string, string> criteria, MultiOptions? options = null)
        : base("multi", instructions, Build(criteria, options ?? new())) { }
    private static JsonObject Build(IReadOnlyDictionary<string, string> criteria, MultiOptions options)
    {
        var fields = new JsonObject { ["criteria"] = Questions.Criteria(criteria, 1, int.MaxValue) };
        if (options.Threshold is double t) { Questions.Finite(t, true); fields["threshold"] = t; }
        if (options.Min is int min) { if (min < 0 || min > criteria.Count) throw new ArgumentException("Invalid min"); fields["min"] = min; }
        if (options.Max is int max) { if (max < 0 || max > criteria.Count) throw new ArgumentException("Invalid max"); fields["max"] = max; }
        if (options.Min > options.Max) throw new ArgumentException("Min exceeds Max");
        return fields;
    }
    /// <inheritdoc/>
    public override MultiResult Read(DecisionResponse response, string id)
    {
        var a = Find(response, id, "multi"); var criteria = ToJson()["criteria"]!.AsObject();
        if (a.Selected is null || a.Selected.Any(k => k is null || !criteria.ContainsKey(k))) throw new BasalException("Invalid selected labels");
        return new(a.Selected, Number(a.Threshold), Number(a.SetConfidence), a.Probabilities) { Extra = a.Extra ?? [] };
    }
}
/// <summary>Minimum expected cost over an explicit matrix.</summary>
public sealed class ActQuestion : QuestionSpec<ActResult>
{
    /// <summary>Create act; every action must give a finite cost for every outcome.</summary>
    public ActQuestion(string instructions, IReadOnlyDictionary<string, Dictionary<string, double>> costs, ActOptions? options = null)
        : base("act", instructions, Build(costs, options ?? new())) { }
    private static JsonObject Build(IReadOnlyDictionary<string, Dictionary<string, double>> costs, ActOptions options)
    {
        ArgumentNullException.ThrowIfNull(costs);
        if (costs.Count == 0) throw new ArgumentException("Act requires costs");
        var criteria = options.Criteria is null ? null : Questions.Criteria(options.Criteria, 2, 10);
        var outcomes = criteria is null ? new HashSet<string> { "true", "false" } : criteria.Select(p => p.Key).ToHashSet();
        var matrix = new JsonObject();
        foreach (var (action, row) in costs)
        {
            Questions.Text(action, "action");
            if (row is null || !outcomes.SetEquals(row.Keys)) throw new ArgumentException("Cost row must cover every outcome exactly");
            var values = new JsonObject(); foreach (var (key, value) in row) { Questions.Finite(value); values[key] = value; }
            matrix[action] = values;
        }
        var fields = Questions.Display(options.Evidence); fields["costs"] = matrix;
        if (criteria is not null) fields["criteria"] = criteria;
        if (options.DeferAction is string defer) { if (!costs.ContainsKey(defer)) throw new ArgumentException("Unknown defer action"); fields["defer_action"] = defer; }
        if (options.MaxError is double error) { Questions.Finite(error, true); fields["max_error"] = error; }
        return fields;
    }
    /// <inheritdoc/>
    public override ActResult Read(DecisionResponse response, string id)
    {
        var a = Find(response, id, "act");
        if (a.Action is null || a.Outcome is null || a.Calibration is not ("validated" or "unvalidated") || a.ExpectedCosts is null
            || !ToJson()["costs"]!.AsObject().ContainsKey(a.Action) || a.ExpectedCosts.Values.Any(v => !double.IsFinite(v)))
            throw new BasalException("Invalid act answer");
        return new(a.Action, a.Outcome, Number(a.Confidence), a.Calibration, a.ExpectedCosts, a.Probabilities)
        { Evidence = a.Evidence ?? [], Extra = a.Extra ?? [], Refused = a.Refused, MaxError = a.MaxError };
    }
}
/// <summary>Convenient factories, also available through using static ITSOL.Basal.Questions.</summary>
public static class Questions
{
    /// <summary>Choose one of 2–10 named options.</summary>
    public static ChoiceQuestion Choice(string instructions, IReadOnlyDictionary<string, string> criteria, ChoiceOptions? options = null) => new(instructions, criteria, options);
    /// <summary>Yes/no probability.</summary>
    public static NoulQuestion Noul(string instructions, EvidenceOptions? options = null) => new(instructions, options);
    /// <summary>Score over 2–10 ordered levels.</summary>
    public static ScoreQuestion Score(string instructions, IReadOnlyList<string> levels, EvidenceOptions? options = null) => new(instructions, levels, options);
    /// <summary>Select labels independently.</summary>
    public static MultiQuestion Multi(string instructions, IReadOnlyDictionary<string, string> criteria, MultiOptions? options = null) => new(instructions, criteria, options);
    /// <summary>Minimum expected cost action.</summary>
    public static ActQuestion Act(string instructions, IReadOnlyDictionary<string, Dictionary<string, double>> costs, ActOptions? options = null) => new(instructions, costs, options);
    /// <summary>Build a question map without declaring dictionary/generic types. IDs must be unique.</summary>
    public static IReadOnlyDictionary<string, QuestionSpec> Create(params (string Id, QuestionSpec Question)[] questions)
    {
        ArgumentNullException.ThrowIfNull(questions);
        var result = new Dictionary<string, QuestionSpec>();
        foreach (var (id, question) in questions)
        {
            Text(id, "question ID"); ArgumentNullException.ThrowIfNull(question);
            if (!result.TryAdd(id, question)) throw new ArgumentException("Duplicate question ID");
        }
        return result;
    }
    /// <summary>Choice from named tuples, with inferred types and default options.</summary>
    public static ChoiceQuestion Choice(string instructions, params (string Key, string Description)[] criteria)
        => new(instructions, Named(criteria));
    /// <summary>Multi from named tuples with default options.</summary>
    public static MultiQuestion Multi(string instructions, params (string Key, string Description)[] criteria)
        => new(instructions, Named(criteria));
    /// <summary>Score from ordered level arguments with default options.</summary>
    public static ScoreQuestion Score(string instructions, params string[] levels) => new(instructions, levels);
    private static Dictionary<string, string> Named((string Key, string Description)[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var result = new Dictionary<string, string>();
        foreach (var (key, value) in values) if (!result.TryAdd(key, value)) throw new ArgumentException("Duplicate criterion key");
        return result;
    }
    internal static void Text(string value, string name) { if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException("Expected nonempty text", name); }
    internal static void Finite(double value, bool probability = false) { if (!double.IsFinite(value) || (probability && (value < 0 || value > 1))) throw new ArgumentException("Expected finite number in range"); }
    internal static JsonObject Display(bool evidence) => evidence ? new JsonObject { ["evidence"] = true } : new JsonObject();
    internal static JsonObject Criteria(IReadOnlyDictionary<string, string> criteria, int min, int max)
    {
        ArgumentNullException.ThrowIfNull(criteria);
        if (criteria.Count < min || criteria.Count > max) throw new ArgumentException("Invalid number of criteria");
        var result = new JsonObject(); foreach (var (key, value) in criteria) { Text(key, "criterion key"); Text(value, "criterion description"); result[key] = value; }
        return result;
    }
}


/// <summary>Checked results keyed by their request IDs.</summary>
public sealed class TypedDecisionResponse
{
    /// <summary>Read-only map of checked result records. Match the result type for kind-specific fields.</summary>
    public IReadOnlyDictionary<string, AnswerResult> Answers { get; }
    /// <summary>Native response retained for extension fields.</summary>
    public DecisionResponse Raw { get; }
    /// <summary>Model actually used.</summary>
    public string Model => Raw.Model;
    /// <summary>Server usage.</summary>
    public Usage Usage => Raw.Usage;
    internal TypedDecisionResponse(DecisionResponse raw, KeyValuePair<string, QuestionSpec>[] questions)
    {
        if (string.IsNullOrEmpty(raw.Model) || raw.Usage is null || raw.Usage.InputTokens < 0 || raw.Usage.OutputTokens < 0)
            throw new BasalException("Invalid decision envelope");
        Raw = raw;
        var answers = new Dictionary<string, AnswerResult>(StringComparer.Ordinal);
        foreach (var (id, question) in questions) answers.Add(id, question.Parse(raw, id));
        Answers = new System.Collections.ObjectModel.ReadOnlyDictionary<string, AnswerResult>(answers);
    }
    /// <summary>Get a checked result by request ID. An unknown ID throws KeyNotFoundException.</summary>
    public AnswerResult Answer(string id) => Answers[id];
}
