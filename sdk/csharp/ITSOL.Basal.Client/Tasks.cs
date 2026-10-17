using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace ITSOL.Basal;

/// <summary>Experimental classification options; certificates apply only to the named task.</summary>
public sealed record ClassifyOptions(bool MultiLabel = false, bool None = false, IReadOnlyList<double>? MaxError = null,
    OptionKeys OptionKeys = OptionKeys.Show, MultiOptions? Selection = null);
/// <summary>Experimental verification options.</summary>
public sealed record VerifyOptions(bool Binary = false, IReadOnlyList<double>? MaxError = null);
/// <summary>Extractive compression options; whole source units may exceed KeepRatio.</summary>
public sealed record CompressOptions(string Unit = "sentence", double? KeepRatio = null, double? Threshold = null, int? MinKeep = null);
/// <summary>A verified source claim.</summary>
public sealed class ClaimResult
{
    /// <summary>supported, contradicted, not_stated or not_supported.</summary>
    [JsonPropertyName("verdict")] public required string Verdict { get; init; }
    /// <summary>Outcome probabilities.</summary>
    [JsonPropertyName("probabilities")] public required Dictionary<string,double> Probabilities { get; init; }
    /// <summary>Maximum probability.</summary>
    [JsonPropertyName("confidence")] public required double Confidence { get; init; }
    /// <summary>Certified task automation; null means unavailable.</summary>
    [JsonPropertyName("automated")] public required Dictionary<string,bool?> Automated { get; init; }
}
/// <summary>One source unit; offsets count Unicode code points, not UTF-16 code units.</summary>
public sealed class CompressionUnit
{
    /// <summary>Inclusive start.</summary>
    [JsonPropertyName("start")] public required int Start { get; init; }
    /// <summary>Exclusive end.</summary>
    [JsonPropertyName("end")] public required int End { get; init; }
    /// <summary>Retention probability.</summary>
    [JsonPropertyName("probability")] public required double Probability { get; init; }
}
/// <summary>Extraction provenance. Span confidence is a product of individually calibrated decisions.</summary>
public sealed class ExtractField
{
    /// <summary>Native value, including explicit null.</summary>
    [JsonPropertyName("value")] public required JsonElement Value { get; init; }
    /// <summary>Decision confidence or line/span product.</summary>
    [JsonPropertyName("confidence")] public required double Confidence { get; init; }
    /// <summary>choice or span.</summary>
    [JsonPropertyName("source")] public required string Source { get; init; }
    /// <summary>validated or per decision.</summary>
    [JsonPropertyName("calibration")] public required string Calibration { get; init; }
    /// <summary>Categorical probabilities.</summary>
    [JsonPropertyName("probabilities")] public Dictionary<string,double>? Probabilities { get; init; }
    /// <summary>Probability the field is present.</summary>
    [JsonPropertyName("present_probability")] public double? PresentProbability { get; init; }
    /// <summary>Verbatim source span.</summary>
    [JsonPropertyName("text")] public string? Text { get; init; }
    /// <summary>Inclusive Unicode code point offset.</summary>
    [JsonPropertyName("start")] public int? Start { get; init; }
    /// <summary>Exclusive Unicode code point offset.</summary>
    [JsonPropertyName("end")] public int? End { get; init; }
    /// <summary>Line selection probability.</summary>
    [JsonPropertyName("line_probability")] public double? LineProbability { get; init; }
    /// <summary>Span selection probability.</summary>
    [JsonPropertyName("span_probability")] public double? SpanProbability { get; init; }
    /// <summary>Numeric conversion diagnostic.</summary>
    [JsonPropertyName("error")] public string? Error { get; init; }
}
/// <summary>Common marker for experimental task results.</summary>
public abstract record TaskResult : AnswerResult
{
    /// <summary>These task interfaces are experimental in upstream 2.1.</summary>
    public bool Experimental => true;
}
/// <summary>Checked classification scores and single/multi selection.</summary>
public sealed record ClassifyResult(IReadOnlyDictionary<string,double> Scores, string? Label, string? TopLabel,
    double? Confidence, IReadOnlyDictionary<string,bool?>? Automated, IReadOnlyList<string>? Labels, double? SetConfidence) : TaskResult;
/// <summary>Checked claim verification.</summary>
public sealed record VerifyResult(IReadOnlyDictionary<string,ClaimResult> Claims, bool AllSupported, IReadOnlyList<string> NotSupported) : TaskResult;
/// <summary>Checked source-preserving compression.</summary>
public sealed record CompressResult(string Compressed, IReadOnlyList<int> Kept, IReadOnlyList<CompressionUnit> Units,
    string Unit, int OriginalChars, int CompressedChars, double Ratio) : TaskResult;
/// <summary>Checked structured extraction with arbitrary JSON values.</summary>
public sealed record ExtractResult(JsonObject Data, IReadOnlyDictionary<string,ExtractField> Fields) : TaskResult;

internal static class TaskChecks
{
    internal static void Probability(double? value) { if(value is not double n||!double.IsFinite(n)||n is <0 or >1)throw new BasalException("Invalid task probability"); }
    internal static void Probabilities(IReadOnlyDictionary<string,double>? values) { if(values is null)throw new BasalException("Missing task probabilities");foreach(var n in values.Values)Probability(n); }
    internal static void Span(int? start,int? end) { if(start is null or <0||end is null||end<start)throw new BasalException("Invalid Unicode code point span"); }
    internal static void Risks(JsonObject q,IReadOnlyList<double>? values) {
        if(values is null)return;
        if(values.Count>64||values.Any(n=>!double.IsFinite(n)||n<=0||n>=1))throw new ArgumentException("Invalid task MaxError");
        q["max_error"]=new JsonArray(values.Select(n=>(JsonNode)JsonValue.Create(n)!).ToArray());
    }
    internal static JsonNode ListOrMap(JsonNode values) {
        ArgumentNullException.ThrowIfNull(values);
        if(values is not (JsonObject or JsonArray)|| (values is JsonObject o?o.Count:((JsonArray)values).Count)==0)throw new ArgumentException("Expected a nonempty object or list");
        return values.DeepClone();
    }
    internal static void Experimental(Answer a) {if(a.Experimental!=true)throw new BasalException("Missing experimental task marker");}
}
/// <summary>Classify into one or several labels.</summary>
public sealed class ClassifyQuestion : QuestionSpec<ClassifyResult>
{
    /// <summary>Create an experimental task; empty instructions use the server default.</summary>
    public ClassifyQuestion(string instructions,JsonNode labels,ClassifyOptions? options=null):base("classify",instructions,Build(labels,options??new())) { }
    private static JsonObject Build(JsonNode labels,ClassifyOptions o) {
        var copy=TaskChecks.ListOrMap(labels);if((copy is JsonObject obj?obj.Count:((JsonArray)copy).Count)>256)throw new ArgumentException("At most 256 labels");
        if(!Enum.IsDefined(o.OptionKeys))throw new ArgumentException("Invalid OptionKeys");
        var q=new JsonObject{["labels"]=copy,["multi_label"]=o.MultiLabel,["none"]=o.None,["option_keys"]=o.OptionKeys==OptionKeys.Hide?"hide":"show"};TaskChecks.Risks(q,o.MaxError);
        var selection=o.Selection??new();
        if(selection.Exclusive is not null||selection.Fallback is not null||selection.FallbackThreshold is not null)throw new ArgumentException("Classify does not support exclusive/fallback");
        var fields=Questions.SelectionOptions(selection,null);if(!o.MultiLabel&&fields.Count>0)throw new ArgumentException("Selection requires MultiLabel");
        foreach(var (k,v) in fields)q[k]=v?.DeepClone();return q;
    }
    /// <inheritdoc/>
    public override ClassifyResult Read(DecisionResponse response,string id) {
        var a=Find(response,id,"classify");TaskChecks.Experimental(a);TaskChecks.Probabilities(a.Scores);
        if(ToJson()["multi_label"]!.GetValue<bool>()) {if(a.Labels is null||a.Labels.Any(k=>k is null||!a.Scores!.ContainsKey(k)))throw new BasalException("Invalid labels");TaskChecks.Probability(a.SetConfidence);}
        else {if(a.Label is null||!a.Scores!.ContainsKey(a.Label)||a.TopLabel!=a.Label||a.Automated is null)throw new BasalException("Invalid label/automation");TaskChecks.Probability(a.Confidence);}
        return new(a.Scores!,a.Label,a.TopLabel,a.Confidence,a.Automated,a.Labels,a.SetConfidence){Extra=a.Extra??[]};
    }
}
/// <summary>Verify source claims with two or three outcomes.</summary>
public sealed class VerifyQuestion : QuestionSpec<VerifyResult>
{
    /// <summary>Create an experimental verification task.</summary>
    public VerifyQuestion(string instructions,JsonNode claims,VerifyOptions? options=null):base("verify",instructions,Build(claims,options??new())) { }
    private static JsonObject Build(JsonNode claims,VerifyOptions o){var q=new JsonObject{["claims"]=TaskChecks.ListOrMap(claims),["binary"]=o.Binary};TaskChecks.Risks(q,o.MaxError);return q;}
    /// <inheritdoc/>
    public override VerifyResult Read(DecisionResponse response,string id){
        var a=Find(response,id,"verify");TaskChecks.Experimental(a);if(a.Claims is null||a.AllSupported is null||a.NotSupported is null||a.NotSupported.Any(k=>k is null||!a.Claims.ContainsKey(k)))throw new BasalException("Invalid verification summary");
        foreach(var c in a.Claims.Values){if(c is null||c.Verdict is not ("supported" or "contradicted" or "not_stated" or "not_supported")||c.Probabilities is null||!c.Probabilities.ContainsKey(c.Verdict)||c.Automated is null)throw new BasalException("Invalid verdict");TaskChecks.Probability(c.Confidence);TaskChecks.Probabilities(c.Probabilities);}
        return new(a.Claims,a.AllSupported.Value,a.NotSupported){Extra=a.Extra??[]};
    }
}
/// <summary>Keep relevant source sentences, lines or paragraphs.</summary>
public sealed class CompressQuestion : QuestionSpec<CompressResult>
{
    /// <summary>Create an experimental extractive compression task.</summary>
    public CompressQuestion(string instructions="",CompressOptions? options=null):base("compress",instructions,Build(options??new())) { }
    private static JsonObject Build(CompressOptions o){
        if(o.Unit is not ("sentence" or "line" or "paragraph"))throw new ArgumentException("Invalid unit");
        var q=new JsonObject{["unit"]=o.Unit};
        if(o.KeepRatio is double r){Questions.Finite(r,true);if(r==0)throw new ArgumentException("KeepRatio must be positive");q["keep_ratio"]=r;}
        if(o.Threshold is double t){Questions.Finite(t,true);q["threshold"]=t;}
        if(o.MinKeep is int n){if(n<0)throw new ArgumentException("Invalid MinKeep");q["min_keep"]=n;}return q;
    }
    /// <inheritdoc/>
    public override CompressResult Read(DecisionResponse response,string id){
        var a=Find(response,id,"compress");TaskChecks.Experimental(a);
        if(a.Compressed is null||a.Units is null||a.Kept is null||a.Unit is not ("sentence" or "line" or "paragraph")||a.OriginalChars is null or <0||a.CompressedChars is null or <0||a.Ratio is not double ratio||!double.IsFinite(ratio)||ratio<0||a.Kept.Any(i=>i<0||i>=a.Units.Count))throw new BasalException("Invalid compression");
        foreach(var u in a.Units){if(u is null)throw new BasalException("Invalid unit");TaskChecks.Span(u.Start,u.End);TaskChecks.Probability(u.Probability);}
        return new(a.Compressed,a.Kept,a.Units,a.Unit,a.OriginalChars.Value,a.CompressedChars.Value,ratio){Extra=a.Extra??[]};
    }
}
/// <summary>Extract categorical values and verbatim source spans.</summary>
public sealed class ExtractQuestion : QuestionSpec<ExtractResult>
{
    /// <summary>Create an experimental extraction task. Schema validation belongs to the server.</summary>
    public ExtractQuestion(string instructions,JsonObject schema):base("extract",instructions,new JsonObject{["schema"]=(schema??throw new ArgumentNullException(nameof(schema))).DeepClone()}) { }
    /// <inheritdoc/>
    public override ExtractResult Read(DecisionResponse response,string id){
        var a=Find(response,id,"extract");TaskChecks.Experimental(a);if(a.Data is null||a.Fields is null)throw new BasalException("Missing extraction data");
        foreach(var f in a.Fields.Values){if(f is null||f.Value.ValueKind==JsonValueKind.Undefined||f.Source is not ("choice" or "span")||f.Calibration is not ("validated" or "per decision"))throw new BasalException("Invalid extraction field");TaskChecks.Probability(f.Confidence);if(f.Probabilities is not null)TaskChecks.Probabilities(f.Probabilities);foreach(var p in new[]{f.PresentProbability,f.LineProbability,f.SpanProbability})if(p is not null)TaskChecks.Probability(p);if(f.Text is not null)TaskChecks.Span(f.Start,f.End);}
        return new(a.Data,a.Fields){Extra=a.Extra??[]};
    }
}
public static partial class Questions
{
    /// <summary>Classify source content.</summary>
    public static ClassifyQuestion Classify(string instructions,JsonNode labels,ClassifyOptions? options=null)=>new(instructions,labels,options);
    /// <summary>Verify source claims.</summary>
    public static VerifyQuestion Verify(string instructions,JsonNode claims,VerifyOptions? options=null)=>new(instructions,claims,options);
    /// <summary>Compress source text without generation.</summary>
    public static CompressQuestion Compress(string instructions="",CompressOptions? options=null)=>new(instructions,options);
    /// <summary>Extract a schema from source content.</summary>
    public static ExtractQuestion Extract(string instructions,JsonObject schema)=>new(instructions,schema);
}
public sealed partial class QuestionBuilder
{
    /// <summary>Create a named classification result field.</summary>
    public AnswerField<ClassifyResult> Classify(string instructions,JsonNode labels,ClassifyOptions? options=null)=>From(ITSOL.Basal.Questions.Classify(instructions,labels,options));
    /// <summary>Create a named verification result field.</summary>
    public AnswerField<VerifyResult> Verify(string instructions,JsonNode claims,VerifyOptions? options=null)=>From(ITSOL.Basal.Questions.Verify(instructions,claims,options));
    /// <summary>Create a named compression result field.</summary>
    public AnswerField<CompressResult> Compress(string instructions="",CompressOptions? options=null)=>From(ITSOL.Basal.Questions.Compress(instructions,options));
    /// <summary>Create a named extraction result field.</summary>
    public AnswerField<ExtractResult> Extract(string instructions,JsonObject schema)=>From(ITSOL.Basal.Questions.Extract(instructions,schema));
}
