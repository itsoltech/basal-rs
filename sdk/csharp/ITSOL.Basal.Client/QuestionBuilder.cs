using System.Text.Json;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace ITSOL.Basal;

// Every field in a request shares one publication point. Failed requests never
// expose partially validated results, even if a factory captured a field.
internal sealed class AnswerCompletion
{
    private TypedDecisionResponse? response;

    internal TAnswer Get<TAnswer>(AnswerBinding binding) where TAnswer : AnswerResult
    {
        if (binding.Prepared)
            throw new InvalidOperationException("Prepared fields describe questions; read answers from the returned response.Answers");
        var completed = Volatile.Read(ref response)
            ?? throw new InvalidOperationException("Answer fields are available only after a successful DecideAsync");
        return (TAnswer)completed.Answer(binding.Id
            ?? throw new InvalidOperationException("Answer fields must be returned as named members of the factory result"));
    }

    internal void Complete(TypedDecisionResponse value) => Volatile.Write(ref response, value);
}

internal sealed class AnswerBinding(AnswerCompletion completion, QuestionSpec question, string? id)
{
    internal AnswerCompletion Completion { get; } = completion;
    internal QuestionSpec Question { get; } = question;
    internal bool ExplicitId { get; } = id is not null;
    internal string? Id { get; private set; } = id;
    internal bool Prepared { get; private set; }
    internal void MarkPrepared() => Prepared = true;

    internal void BindName(string name, string parameterName)
    {
        if (ExplicitId) return;
        if (Id is not null && Id != name)
            throw new ArgumentException("Create a new answer field for each member; question definitions may be reused", parameterName);
        Id = name;
    }
}

internal interface IAnswerField
{
    AnswerBinding Binding { get; }
    object Copy(AnswerBinding binding);
}

/// <summary>A request-bound answer field. Read after successfully awaiting DecideAsync.</summary>
public class AnswerField<TAnswer> : IAnswerField where TAnswer : AnswerResult
{
    private readonly AnswerBinding binding;
    internal AnswerField(AnswerBinding binding) => this.binding = binding;
    AnswerBinding IAnswerField.Binding => binding;
    object IAnswerField.Copy(AnswerBinding newBinding) => Copy(newBinding);
    internal virtual AnswerField<TAnswer> Copy(AnswerBinding newBinding) => new(newBinding);
    /// <summary>The checked record for this request and ID; throws before successful completion.</summary>
    public TAnswer Result => binding.Completion.Get<TAnswer>(binding);
    /// <summary>Supporting spans, empty when absent.</summary>
    public IReadOnlyList<EvidenceSpan> Evidence => Result.Evidence;
    /// <summary>Additional native result fields.</summary>
    public IReadOnlyDictionary<string, JsonElement> Extra => Result.Extra;
}

/// <summary>A named, checked choice answer.</summary>
public sealed class ChoiceField : AnswerField<ChoiceResult>
{
    internal ChoiceField(AnswerBinding binding) : base(binding) { }
    internal override ChoiceField Copy(AnswerBinding binding) => new(binding);
    /// <summary>The selected criterion key.</summary>
    public string Choice => Result.Choice;
    /// <summary>Native confidence.</summary>
    public double Confidence => Result.Confidence;
    /// <summary>Criterion probabilities.</summary>
    public IReadOnlyDictionary<string, double> Probabilities => Result.Probabilities;
}

/// <summary>A named, checked yes/no answer.</summary>
public sealed class NoulField : AnswerField<NoulResult>
{
    internal NoulField(AnswerBinding binding) : base(binding) { }
    internal override NoulField Copy(AnswerBinding binding) => new(binding);
    /// <summary>Probability of yes.</summary>
    public double Noul => Result.Noul;
    /// <summary>Native confidence.</summary>
    public double Confidence => Result.Confidence;
    /// <summary>Outcome probabilities.</summary>
    public IReadOnlyDictionary<string, double> Probabilities => Result.Probabilities;
}

/// <summary>A named, checked score answer.</summary>
public sealed class ScoreField : AnswerField<ScoreResult>
{
    internal ScoreField(AnswerBinding binding) : base(binding) { }
    internal override ScoreField Copy(AnswerBinding binding) => new(binding);
    /// <summary>The native score.</summary>
    public double Score => Result.Score;
    /// <summary>Native confidence.</summary>
    public double Confidence => Result.Confidence;
    /// <summary>Ordered score labels.</summary>
    public IReadOnlyDictionary<string, string> Legend => Result.Legend;
    /// <summary>Level probabilities.</summary>
    public IReadOnlyDictionary<string, double> Probabilities => Result.Probabilities;
}

/// <summary>A named, checked multi-label answer.</summary>
public sealed class MultiField : AnswerField<MultiResult>
{
    internal MultiField(AnswerBinding binding) : base(binding) { }
    internal override MultiField Copy(AnswerBinding binding) => new(binding);
    /// <summary>Selected criterion keys.</summary>
    public IReadOnlyList<string> Selected => Result.Selected;
    /// <summary>Selection threshold.</summary>
    public double? Threshold => Result.Threshold;
    /// <summary>Selection objective.</summary>
    public string Objective => Result.Objective;
    /// <summary>Expected objective score.</summary>
    public double? ExpectedScore => Result.ExpectedScore;
    /// <summary>Constraint diagnostics, when reported.</summary>
    public JsonElement? ConstraintsApplied => Result.ConstraintsApplied;
    /// <summary>Native set confidence.</summary>
    public double SetConfidence => Result.SetConfidence;
    /// <summary>Independent label probabilities.</summary>
    public IReadOnlyDictionary<string, double> Probabilities => Result.Probabilities;
}

/// <summary>A named, checked minimum-cost answer. Reading it never executes the action.</summary>
public sealed class ActField : AnswerField<ActResult>
{
    internal ActField(AnswerBinding binding) : base(binding) { }
    internal override ActField Copy(AnswerBinding binding) => new(binding);
    /// <summary>Selected action key.</summary>
    public string Action => Result.Action;
    /// <summary>Selected outcome key.</summary>
    public string Outcome => Result.Outcome;
    /// <summary>Native confidence.</summary>
    public double Confidence => Result.Confidence;
    /// <summary>Native calibration status.</summary>
    public string Calibration => Result.Calibration;
    /// <summary>Expected cost per action.</summary>
    public IReadOnlyDictionary<string, double> ExpectedCosts => Result.ExpectedCosts;
    /// <summary>Outcome probabilities.</summary>
    public IReadOnlyDictionary<string, double> Probabilities => Result.Probabilities;
    /// <summary>Optional native refusal explanation.</summary>
    public string? Refused => Result.Refused;
    /// <summary>Optional native certified error target warning.</summary>
    public string? MaxError => Result.MaxError;
    /// <summary>Whether the automatic action was deferred, when reported.</summary>
    public bool? Deferred => Result.Deferred;
}

/// <summary>A named, checked tool routing result.</summary>
public sealed class ToolField : AnswerField<ToolResult>
{
    internal ToolField(AnswerBinding binding):base(binding) { }
    internal override ToolField Copy(AnswerBinding binding)=>new(binding);
    /// <summary>Selected single tool.</summary>
    public string? Tool=>Result.Tool;
    /// <summary>Whether an actual tool was selected.</summary>
    public bool Call=>Result.Call;
    /// <summary>Selected tool set.</summary>
    public IReadOnlyList<string> Tools=>Result.Tools;
    /// <summary>Ranked candidates.</summary>
    public IReadOnlyList<ToolRanking> Ranking=>Result.Ranking;
    /// <summary>Complete routing probabilities.</summary>
    public IReadOnlyDictionary<string,double> Probabilities=>Result.Probabilities;
    /// <summary>Single-tool confidence, null in multi mode.</summary>
    public double? Confidence=>Result.Confidence;
    /// <summary>Multi-tool selection score and constraints.</summary>
    public MultiResult? Selection=>Result.Selection;
    /// <summary>Whether the certified single-tool decision was deferred, when reported.</summary>
    public bool? Deferred=>Result.Deferred;
    /// <summary>Refused tool on deferral.</summary>
    public string? Refused=>Result.Refused;
    /// <summary>Optional native certified error target warning.</summary>
    public string? MaxError=>Result.MaxError;
}
/// <summary>Create typed question and answer fields. Factory methods infer IDs from the returned object's member names.
/// Use only inside a synchronous Prepare or DecideAsync factory; builders are not thread-safe.</summary>
public sealed class QuestionBuilder
{
    private const string SynchronousPreparationRequired = "Prepare requires a synchronous factory. Load asynchronous inputs before preparing questions.";
    private readonly Dictionary<string, QuestionSpec> questions = new(StringComparer.Ordinal);
    private readonly HashSet<string> explicitIds = new(StringComparer.Ordinal);
    private readonly List<AnswerBinding> bindings = [];
    private readonly AnswerCompletion completion = new();
    private bool closed;
    internal QuestionBuilder() { }
    internal IReadOnlyDictionary<string, QuestionSpec> Questions => questions;
    internal void Close() => closed = true;
    internal void Complete(TypedDecisionResponse response) => completion.Complete(response);

    /// <summary>Prepare reusable question fields once, after loading their inputs. The returned shape
    /// can be passed directly to DecideAsync, or selected into a new anonymous object. Its fields
    /// remain definitions; each decision receives fresh, checked answer fields.</summary>
    public static TQuestions Prepare<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties | DynamicallyAccessedMemberTypes.PublicFields)] TQuestions>(
        Func<QuestionBuilder, TQuestions> createQuestions) where TQuestions : notnull
    {
        ArgumentNullException.ThrowIfNull(createQuestions);
        var builder = new QuestionBuilder();
        TQuestions questions;
        try { questions = createQuestions(builder); }
        finally { builder.Close(); }
        if (questions is null) throw new ArgumentException("The question factory must return an object", nameof(createQuestions));
        builder.BindShape(questions, nameof(createQuestions));
        if (builder.bindings.Count == 0) throw new ArgumentException("Prepare requires at least one question", nameof(createQuestions));
        foreach (var binding in builder.bindings) binding.MarkPrepared();
        return questions;
    }

    /// <summary>Asynchronous preparation factories are unsupported; load their inputs first.</summary>
    [Obsolete(SynchronousPreparationRequired, error: true)]
    public static TQuestions Prepare<TQuestions>(Func<QuestionBuilder, Task<TQuestions>> createQuestions) where TQuestions : notnull
        => throw new ArgumentException(SynchronousPreparationRequired, nameof(createQuestions));
    /// <summary>Asynchronous preparation factories are unsupported; load their inputs first.</summary>
    [Obsolete(SynchronousPreparationRequired, error: true)]
    public static TQuestions Prepare<TQuestions>(Func<QuestionBuilder, ValueTask<TQuestions>> createQuestions) where TQuestions : notnull
        => throw new ArgumentException(SynchronousPreparationRequired, nameof(createQuestions));
    /// <summary>Preparation must return its question shape synchronously.</summary>
    [Obsolete(SynchronousPreparationRequired, error: true)]
    public static object Prepare(Func<QuestionBuilder, Task> createQuestions)
        => throw new ArgumentException(SynchronousPreparationRequired, nameof(createQuestions));
    /// <summary>Preparation must return its question shape synchronously.</summary>
    [Obsolete(SynchronousPreparationRequired, error: true)]
    public static object Prepare(Func<QuestionBuilder, ValueTask> createQuestions)
        => throw new ArgumentException(SynchronousPreparationRequired, nameof(createQuestions));

    internal object CopyPrepared(IAnswerField field)
    {
        if (!field.Binding.Prepared)
            throw new ArgumentException("Use fields created by QuestionBuilder.Prepare, not fields from a decision response", "questions");
        var binding = field.Binding;
        if (binding.ExplicitId && binding.Id is { } id && explicitIds.Contains(id))
            throw new ArgumentException("An explicit question ID cannot be selected more than once", "questions");
        return field.Copy(Register(binding.Question, binding.ExplicitId ? binding.Id : null));
    }

    private AnswerBinding Register(QuestionSpec question, string? id = null)
    {
        if (closed) throw new InvalidOperationException("The question builder is closed");
        ArgumentNullException.ThrowIfNull(question);
        if (id is not null)
        {
            ITSOL.Basal.Questions.Text(id, "question ID");
            if (!explicitIds.Add(id)) throw new ArgumentException("Duplicate question ID", nameof(id));
        }
        var binding = new AnswerBinding(completion, question, id);
        bindings.Add(binding);
        return binding;
    }

    private AnswerBinding RegisterExplicit(string id, QuestionSpec question)
    {
        ITSOL.Basal.Questions.Text(id, "question ID");
        return Register(question, id);
    }

    internal void BindShape<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties | DynamicallyAccessedMemberTypes.PublicFields)] TAnswers>(TAnswers answers,
        string parameterName = "createAnswers")
        where TAnswers : notnull
    {
        if (typeof(ITuple).IsAssignableFrom(typeof(TAnswers)) && bindings.Any(binding => !binding.ExplicitId))
            throw new ArgumentException("Tuple element names are not available at runtime; use an anonymous object or class for inferred IDs", parameterName);
        // Inspect only typed answer members. Unrelated getters are never invoked.
        // Build the wire map in factory creation order, independent of reflection order.
        foreach (var property in typeof(TAnswers).GetProperties(BindingFlags.Instance | BindingFlags.Public))
        {
            if (!typeof(IAnswerField).IsAssignableFrom(property.PropertyType)) continue;
            if (property.GetMethod is null || !property.GetMethod.IsPublic || property.GetIndexParameters().Length != 0)
                throw new ArgumentException("Answer fields must be readable, non-indexed public members", parameterName);
            object? value;
            try { value = property.GetValue(answers); }
            catch (TargetInvocationException error)
            {
                throw new ArgumentException("Could not read an answer field from the factory result", parameterName, error.InnerException ?? error);
            }
            BindMember(property.Name, value, parameterName);
        }
        foreach (var field in typeof(TAnswers).GetFields(BindingFlags.Instance | BindingFlags.Public))
        {
            if (typeof(IAnswerField).IsAssignableFrom(field.FieldType)) BindMember(field.Name, field.GetValue(answers), parameterName);
        }
        foreach (var binding in bindings)
        {
            var id = binding.Id ?? throw new ArgumentException("Return each inferred answer field as a public member of the factory result", parameterName);
            if (!questions.TryAdd(id, binding.Question))
                throw new ArgumentException("Duplicate question ID in the question shape", parameterName);
        }
    }

    private void BindMember(string name, object? value, string parameterName)
    {
        if (value is not IAnswerField field || !ReferenceEquals(field.Binding.Completion, completion))
            throw new ArgumentException("Answer members must be non-null fields created by this factory's builder", parameterName);
        field.Binding.BindName(name, parameterName);
    }

    /// <summary>Create a choice field; its request ID is the returned member's exact name.</summary>
    public ChoiceField Choice(string instructions, IReadOnlyDictionary<string, string> criteria, ChoiceOptions? options = null)
        => Choice(ITSOL.Basal.Questions.Choice(instructions, criteria, options));
    /// <summary>Create a choice from named options.</summary>
    public ChoiceField Choice(string instructions, params (string Key, string Description)[] criteria)
        => Choice(ITSOL.Basal.Questions.Choice(instructions, criteria));
    /// <summary>Reuse a choice definition in a fresh named answer field.</summary>
    public ChoiceField Choice(ChoiceQuestion question) => new(Register(question));
    /// <summary>Create a yes/no field; its request ID is the returned member's exact name.</summary>
    public NoulField Noul(string instructions, EvidenceOptions? options = null)
        => Noul(ITSOL.Basal.Questions.Noul(instructions, options));
    /// <summary>Reuse a yes/no definition in a fresh named answer field.</summary>
    public NoulField Noul(NoulQuestion question) => new(Register(question));
    /// <summary>Create a score field from ordered levels.</summary>
    public ScoreField Score(string instructions, IReadOnlyList<string> levels, EvidenceOptions? options = null)
        => Score(ITSOL.Basal.Questions.Score(instructions, levels, options));
    /// <summary>Create a score field from ordered level arguments.</summary>
    public ScoreField Score(string instructions, params string[] levels)
        => Score(ITSOL.Basal.Questions.Score(instructions, levels));
    /// <summary>Reuse a score definition in a fresh named answer field.</summary>
    public ScoreField Score(ScoreQuestion question) => new(Register(question));
    /// <summary>Create a multi-label field with typed options.</summary>
    public MultiField Multi(string instructions, IReadOnlyDictionary<string, string> criteria, MultiOptions? options = null)
        => Multi(ITSOL.Basal.Questions.Multi(instructions, criteria, options));
    /// <summary>Create a multi-label field from named options.</summary>
    public MultiField Multi(string instructions, params (string Key, string Description)[] criteria)
        => Multi(ITSOL.Basal.Questions.Multi(instructions, criteria));
    /// <summary>Reuse a multi-label definition in a fresh named answer field.</summary>
    public MultiField Multi(MultiQuestion question) => new(Register(question));
    /// <summary>Create a minimum-cost action field.</summary>
    public ActField Act(string instructions, IReadOnlyDictionary<string, Dictionary<string, double>> costs, ActOptions? options = null)
        => Act(ITSOL.Basal.Questions.Act(instructions, costs, options));
    /// <summary>Reuse an action definition in a fresh named answer field.</summary>
    public ActField Act(ActQuestion question) => new(Register(question));
    /// <summary>Create a tool routing field.</summary>
    public ToolField Tool(string instructions,IReadOnlyList<System.Text.Json.Nodes.JsonObject> tools,ToolOptions? options=null)=>Tool(ITSOL.Basal.Questions.Tool(instructions,tools,options));
    /// <summary>Reuse a tool definition in a fresh named field.</summary>
    public ToolField Tool(ToolQuestion question)=>new(Register(question));
    /// <summary>Create a named field for a generic definition; read its Result property.</summary>
    public AnswerField<TAnswer> From<TAnswer>(QuestionSpec<TAnswer> question) where TAnswer : AnswerResult => new(Register(question));

    /// <summary>Bind a choice definition. Reuse the definition under different IDs as needed.</summary>
    public ChoiceField Add(string id, ChoiceQuestion question) => new(RegisterExplicit(id, question));
    /// <summary>Bind a yes/no definition.</summary>
    public NoulField Add(string id, NoulQuestion question) => new(RegisterExplicit(id, question));
    /// <summary>Bind a score definition.</summary>
    public ScoreField Add(string id, ScoreQuestion question) => new(RegisterExplicit(id, question));
    /// <summary>Bind a multi-label definition.</summary>
    public MultiField Add(string id, MultiQuestion question) => new(RegisterExplicit(id, question));
    /// <summary>Bind an action definition.</summary>
    public ActField Add(string id, ActQuestion question) => new(RegisterExplicit(id, question));
    /// <summary>Bind a tool routing definition.</summary>
    public ToolField Add(string id, ToolQuestion question) => new(RegisterExplicit(id, question));
    /// <summary>Bind a generic definition, with typed access through the field's Result property.</summary>
    public AnswerField<TAnswer> Add<TAnswer>(string id, QuestionSpec<TAnswer> question) where TAnswer : AnswerResult
        => new(RegisterExplicit(id, question));
}

/// <summary>Checked results retaining the anonymous-object or class shape built by the caller.</summary>
public sealed class TypedDecisionResponse<TAnswers> where TAnswers : notnull
{
    private readonly TypedDecisionResponse response;
    internal TypedDecisionResponse(TypedDecisionResponse response, TAnswers answers)
    {
        this.response = response;
        Answers = answers;
    }
    /// <summary>The caller's typed answer shape, populated for this request.</summary>
    public TAnswers Answers { get; }
    /// <summary>Native response and extensions.</summary>
    public DecisionResponse Raw => response.Raw;
    /// <summary>Model actually used.</summary>
    public string Model => response.Model;
    /// <summary>Server usage.</summary>
    public Usage Usage => response.Usage;
}
