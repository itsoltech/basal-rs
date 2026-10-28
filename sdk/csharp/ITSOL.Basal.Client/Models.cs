using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace ITSOL.Basal;

/// <summary>Native Basal question. The server validates fields and model capabilities.</summary>
public sealed class Question
{
    /// <summary>choice (default), noul, score, multi, act, tool, classify, verify, compress or extract.</summary>
    [JsonPropertyName("type")] public string? Type { get; init; }
    /// <summary>Instruction text or another native JSON value.</summary>
    [JsonPropertyName("instructions")] public JsonNode? Instructions { get; init; }
    /// <summary>Ordered criterion object or array. Preserve insertion order.</summary>
    [JsonPropertyName("criteria")] public JsonNode? Criteria { get; init; }
    /// <summary>Native alias for score criteria.</summary>
    [JsonPropertyName("levels")] public JsonNode? Levels { get; init; }
    /// <summary>show or hide.</summary>
    [JsonPropertyName("option_keys")] public string? OptionKeys { get; init; }
    /// <summary>Request supporting spans (unsupported for multi).</summary>
    [JsonPropertyName("evidence")] public bool? Evidence { get; init; }
    /// <summary>Multi label selection threshold.</summary>
    [JsonPropertyName("threshold")] public double? Threshold { get; init; }
    /// <summary>Minimum selected labels.</summary>
    [JsonPropertyName("min")] public int? Min { get; init; }
    /// <summary>Maximum selected labels.</summary>
    [JsonPropertyName("max")] public int? Max { get; init; }
    /// <summary>Act base: noul, choice or score.</summary>
    [JsonPropertyName("base")] public string? Base { get; init; }
    /// <summary>Act cost matrix or short form with wrong/defer costs.</summary>
    [JsonPropertyName("costs")] public JsonNode? Costs { get; init; }
    /// <summary>Act defer action name.</summary>
    [JsonPropertyName("defer_action")] public string? DeferAction { get; init; }
    /// <summary>Certified error target: a number for primitives, a number or array for classify/verify.</summary>
    [JsonPropertyName("max_error")] public JsonNode? MaxError { get; init; }
    /// <summary>threshold, exact, iou or f1.</summary>
    [JsonPropertyName("objective")] public string? Objective { get; init; }
    /// <summary>Mutually exclusive labels.</summary>
    [JsonPropertyName("exclusive")] public IReadOnlyList<string>? Exclusive { get; init; }
    /// <summary>Fallback label.</summary>
    [JsonPropertyName("fallback")] public string? Fallback { get; init; }
    /// <summary>Explicit fallback threshold.</summary>
    [JsonPropertyName("fallback_threshold")] public double? FallbackThreshold { get; init; }
    /// <summary>OpenAI, MCP or Anthropic definition objects.</summary>
    [JsonPropertyName("tools")] public IReadOnlyList<JsonObject>? Tools { get; init; }
    /// <summary>answer_directly, ask_user or none.</summary>
    [JsonPropertyName("extra")] public IReadOnlyList<string>? ExtraOptions { get; init; }
    /// <summary>Number of ranked candidates.</summary>
    [JsonPropertyName("rank")] public int? Rank { get; init; }
    /// <summary>Tool-description truncation cap; omitted keeps the server default (300), zero disables truncation.</summary>
    [JsonPropertyName("description_chars")] public int? DescriptionChars { get; init; }
    /// <summary>Select a set of tools.</summary>
    [JsonPropertyName("multi")] public bool? Multi { get; init; }
    /// <summary>Task labels.</summary>
    [JsonPropertyName("labels")] public JsonNode? Labels { get; init; }
    /// <summary>Multi-label classification.</summary>
    [JsonPropertyName("multi_label")] public bool? MultiLabel { get; init; }
    /// <summary>Add the none-of-the-above category.</summary>
    [JsonPropertyName("none")] public bool? None { get; init; }
    /// <summary>Claims to verify.</summary>
    [JsonPropertyName("claims")] public JsonNode? Claims { get; init; }
    /// <summary>Binary verification.</summary>
    [JsonPropertyName("binary")] public bool? Binary { get; init; }
    /// <summary>Compression segmentation.</summary>
    [JsonPropertyName("unit")] public string? Unit { get; init; }
    /// <summary>Compression retention target.</summary>
    [JsonPropertyName("keep_ratio")] public double? KeepRatio { get; init; }
    /// <summary>Minimum retained units.</summary>
    [JsonPropertyName("min_keep")] public int? MinKeep { get; init; }
    /// <summary>Extraction schema.</summary>
    [JsonPropertyName("schema")] public JsonObject? Schema { get; init; }
    /// <summary>Additional native fields. Do not repeat named fields above.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Request to POST /v1/basal.</summary>
public sealed class DecisionRequest
{
    /// <summary>Non-null native JSON state.</summary>
    [JsonPropertyName("state")] public required JsonNode State { get; init; }
    /// <summary>Question IDs mapped to native questions.</summary>
    [JsonPropertyName("questions")] public required Dictionary<string, Question> Questions { get; init; }
    /// <summary>Exact basal-rs model name; omitted uses the server default.</summary>
    [JsonPropertyName("model")] public string? Model { get; init; }
    /// <summary>off or auto (basal-1.5).</summary>
    [JsonPropertyName("facts")] public string? Facts { get; init; }
    /// <summary>off, auto or force. Null is omitted; use raw JSON to send an explicit wire null.</summary>
    [JsonPropertyName("think")] public string? Think { get; init; }
    /// <summary>Include generated notes, when explicitly requested.</summary>
    [JsonPropertyName("think_text")] public bool? ThinkText { get; init; }
    /// <summary>Open policy string; omitted/null inherits the backend default.</summary>
    [JsonPropertyName("early_exit")] public string? EarlyExit { get; init; }
    /// <summary>Explicit prompt language, pl or en.</summary>
    [JsonPropertyName("lang")] public string? Lang { get; init; }
    /// <summary>Inline media in processing order.</summary>
    [JsonPropertyName("media")] public IReadOnlyList<MediaInput>? Media { get; init; }
    /// <summary>Image detail, default or high.</summary>
    [JsonPropertyName("detail")] public string? Detail { get; init; }
    /// <summary>Additional native fields. Do not repeat named fields above.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Supporting span with offsets in Unicode characters, not UTF-16 code units.</summary>
public sealed class EvidenceSpan
{
    /// <summary>Verbatim span text.</summary>
    [JsonPropertyName("text")] public required string Text { get; init; }
    /// <summary>Inclusive start.</summary>
    [JsonPropertyName("start")] public required int Start { get; init; }
    /// <summary>Exclusive end.</summary>
    [JsonPropertyName("end")] public required int End { get; init; }
    /// <summary>Span probability.</summary>
    [JsonPropertyName("probability")] public required double Probability { get; init; }
}

/// <summary>Native answer; fields for other question types are null.</summary>
public sealed class Answer
{
    /// <summary>choice, noul, score, multi, act, tool, classify, verify, compress or extract.</summary>
    [JsonPropertyName("type")] public required string Type { get; init; }
    /// <summary>Native probabilities; independent P(yes) values for multi.</summary>
    [JsonPropertyName("probabilities")] public Dictionary<string, double> Probabilities { get; init; } = [];
    /// <summary>Max probability for choice/noul/score/act; multi uses set_confidence.</summary>
    [JsonPropertyName("confidence")] public double? Confidence { get; init; }
    /// <summary>Chosen choice key.</summary>
    [JsonPropertyName("choice")] public string? Choice { get; init; }
    /// <summary>P(yes) for noul.</summary>
    [JsonPropertyName("noul")] public double? Noul { get; init; }
    /// <summary>Weighted zero-based score.</summary>
    [JsonPropertyName("score")] public double? Score { get; init; }
    /// <summary>Displayed level texts for score.</summary>
    [JsonPropertyName("legend")] public Dictionary<string, string>? Legend { get; init; }
    /// <summary>Selected multi keys.</summary>
    [JsonPropertyName("selected")] public List<string>? Selected { get; init; }
    /// <summary>Multi threshold.</summary>
    [JsonPropertyName("threshold")] public double? Threshold { get; init; }
    /// <summary>Probability of exactly the selected multi set.</summary>
    [JsonPropertyName("set_confidence")] public double? SetConfidence { get; init; }
    /// <summary>Chosen act action.</summary>
    [JsonPropertyName("action")] public string? Action { get; init; }
    /// <summary>Most probable act outcome.</summary>
    [JsonPropertyName("answer")] public string? Outcome { get; init; }
    /// <summary>Expected cost by act action.</summary>
    [JsonPropertyName("expected_costs")] public Dictionary<string, double>? ExpectedCosts { get; init; }
    /// <summary>validated, unvalidated or text.</summary>
    [JsonPropertyName("calibration")] public string? Calibration { get; init; }
    /// <summary>Refused automatic act action.</summary>
    [JsonPropertyName("refused")] public string? Refused { get; init; }
    /// <summary>Message when the requested certified threshold is unavailable.</summary>
    [JsonPropertyName("max_error")] public string? MaxError { get; init; }
    /// <summary>Whether the automatic decision was deferred.</summary>
    [JsonPropertyName("deferred")] public bool? Deferred { get; init; }
    /// <summary>Selection objective.</summary>
    [JsonPropertyName("objective")] public string? Objective { get; init; }
    /// <summary>Expected selection score.</summary>
    [JsonPropertyName("expected_score")] public double? ExpectedScore { get; init; }
    /// <summary>Constraint diagnostics.</summary>
    [JsonPropertyName("constraints_applied")] public JsonElement? ConstraintsApplied { get; init; }
    /// <summary>Selected single tool, null on deferral.</summary>
    [JsonPropertyName("tool")] public string? Tool { get; init; }
    /// <summary>Selected tools in multi mode.</summary>
    [JsonPropertyName("tools")] public IReadOnlyList<string>? Tools { get; init; }
    /// <summary>Whether an actual tool was selected.</summary>
    [JsonPropertyName("call")] public bool? Call { get; init; }
    /// <summary>Whether this is a multi-tool result.</summary>
    [JsonPropertyName("multi")] public bool? Multi { get; init; }
    /// <summary>Ranked tool candidates.</summary>
    [JsonPropertyName("ranking")] public IReadOnlyList<ToolRanking>? Ranking { get; init; }
    /// <summary>Optional supporting spans.</summary>
    [JsonPropertyName("evidence")] public List<EvidenceSpan>? Evidence { get; init; }
    /// <summary>Experimental task marker.</summary>
    [JsonPropertyName("experimental")] public bool? Experimental { get; init; }
    /// <summary>Single classification label.</summary>
    [JsonPropertyName("label")] public string? Label { get; init; }
    /// <summary>Top classification label.</summary>
    [JsonPropertyName("top_label")] public string? TopLabel { get; init; }
    /// <summary>Selected classification labels.</summary>
    [JsonPropertyName("labels")] public IReadOnlyList<string>? Labels { get; init; }
    /// <summary>Classification scores.</summary>
    [JsonPropertyName("scores")] public Dictionary<string,double>? Scores { get; init; }
    /// <summary>Task-specific certified automation; null means unavailable.</summary>
    [JsonPropertyName("automated")] public Dictionary<string,bool?>? Automated { get; init; }
    /// <summary>Per-claim verification.</summary>
    [JsonPropertyName("claims")] public Dictionary<string,ClaimResult>? Claims { get; init; }
    /// <summary>Whether every claim is supported.</summary>
    [JsonPropertyName("all_supported")] public bool? AllSupported { get; init; }
    /// <summary>Flagged claim keys.</summary>
    [JsonPropertyName("not_supported")] public IReadOnlyList<string>? NotSupported { get; init; }
    /// <summary>Compressed source text.</summary>
    [JsonPropertyName("compressed")] public string? Compressed { get; init; }
    /// <summary>Kept source unit indices.</summary>
    [JsonPropertyName("kept")] public IReadOnlyList<int>? Kept { get; init; }
    /// <summary>Original source units.</summary>
    [JsonPropertyName("units")] public IReadOnlyList<CompressionUnit>? Units { get; init; }
    /// <summary>Compression segmentation.</summary>
    [JsonPropertyName("unit")] public string? Unit { get; init; }
    /// <summary>Source Unicode code point count.</summary>
    [JsonPropertyName("original_chars")] public int? OriginalChars { get; init; }
    /// <summary>Compressed Unicode code point count.</summary>
    [JsonPropertyName("compressed_chars")] public int? CompressedChars { get; init; }
    /// <summary>Character retention ratio.</summary>
    [JsonPropertyName("ratio")] public double? Ratio { get; init; }
    /// <summary>Nested extracted data.</summary>
    [JsonPropertyName("data")] public JsonObject? Data { get; init; }
    /// <summary>Extraction provenance.</summary>
    [JsonPropertyName("fields")] public Dictionary<string,ExtractField>? Fields { get; init; }
    /// <summary>Future answer fields, including raw think and meta objects with all nested keys.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Server-reported usage.</summary>
public sealed class Usage
{
    /// <summary>Prompt tokens for option orders.</summary>
    [JsonPropertyName("input_tokens")] public required long InputTokens { get; init; }
    /// <summary>Generated notes/stop/EOS tokens may contribute, including a skipped second prompt.</summary>
    [JsonPropertyName("output_tokens")] public required long OutputTokens { get; init; }
    /// <summary>Optional question count.</summary>
    [JsonPropertyName("questions")] public long? Questions { get; init; }
    /// <summary>Optional branch count; not multiplied by option orders.</summary>
    [JsonPropertyName("branches")] public long? Branches { get; init; }
    /// <summary>Server-side latency in milliseconds.</summary>
    [JsonPropertyName("latency_ms")] public double? LatencyMs { get; init; }
    /// <summary>Longest executed full prompt.</summary>
    [JsonPropertyName("longest_prompt_tokens")] public long? LongestPromptTokens { get; init; }
    /// <summary>Branches executed in the second option order.</summary>
    [JsonPropertyName("second_order_branches")] public long? SecondOrderBranches { get; init; }
    /// <summary>Media token accounting.</summary>
    [JsonPropertyName("media")] public MediaUsage? Media { get; init; }
    /// <summary>Server scheduling lane, when reported.</summary>
    [JsonPropertyName("lane"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? Lane { get; init; }
    /// <summary>Raw truncation accounting: state/head/tail/dropped tokens and head_share; retains nested keys.</summary>
    [JsonPropertyName("truncated"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public JsonElement? Truncated { get; init; }
    /// <summary>Thinking question count; absent when the server does not report thinking.</summary>
    [JsonPropertyName("think_questions"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public long? ThinkQuestions { get; init; }
    /// <summary>Future usage fields.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Native inference result.</summary>
public sealed class DecisionResponse
{
    /// <summary>Model actually used.</summary>
    [JsonPropertyName("model")] public required string Model { get; init; }
    /// <summary>Native answers keyed by question ID.</summary>
    [JsonPropertyName("answers")] public required Dictionary<string, Answer> Answers { get; init; }
    /// <summary>Server-reported usage.</summary>
    [JsonPropertyName("usage")] public required Usage Usage { get; init; }
    /// <summary>Future envelope fields.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Model discovery entry.</summary>
public sealed class ModelMetadata
{
    /// <summary>Exact model name.</summary>
    [JsonPropertyName("name")] public required string Name { get; init; }
    /// <summary>Prompt protocol version.</summary>
    [JsonPropertyName("prompt_format")] public int? PromptFormat { get; init; }
    /// <summary>Fixed numeric or adaptive string order policy.</summary>
    [JsonPropertyName("orders")] public JsonNode? Orders { get; init; }
    /// <summary>Enabled media kinds.</summary>
    [JsonPropertyName("media")] public IReadOnlyList<string>? Media { get; init; }
    /// <summary>Temperature by primitive.</summary>
    [JsonPropertyName("temperature_per_prim")] public Dictionary<string,double>? TemperaturePerPrim { get; init; }
    /// <summary>Effective prompt layout, when reported.</summary>
    [JsonPropertyName("layout"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? Layout { get; init; }
    /// <summary>Calibration warning, when reported.</summary>
    [JsonPropertyName("calibration_warning"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? CalibrationWarning { get; init; }
    /// <summary>Raw thinking default, available modes, limits and independent calibration identifier.</summary>
    [JsonPropertyName("think"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public JsonElement? Think { get; init; }
    /// <summary>Available policy strings; an empty inventory is meaningful.</summary>
    [JsonPropertyName("early_exit"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public IReadOnlyList<string>? EarlyExit { get; init; }
    /// <summary>Backend default policy, when heads are enabled.</summary>
    [JsonPropertyName("early_exit_default"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? EarlyExitDefault { get; init; }
    /// <summary>Native metadata.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>Model discovery result.</summary>
public sealed class ModelsResponse
{
    /// <summary>Served models.</summary>
    [JsonPropertyName("models")] public required List<ModelMetadata> Models { get; init; }
    /// <summary>Future discovery fields.</summary>
    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; init; }
}
