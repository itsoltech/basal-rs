using System.Text.Json.Serialization;

namespace ITSOL.Basal;

/// <summary>One inline image or audio input. URLs are never fetched by this SDK.</summary>
public sealed record MediaInput(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("data")] string Data);
/// <summary>Inline media helpers; no file I/O or URL fetching.</summary>
public static class Media
{
    /// <summary>Encode image bytes; server validates codec and limits.</summary>
    public static MediaInput Image(ReadOnlySpan<byte> data)=>new("image",Convert.ToBase64String(data));
    /// <summary>Use image base64 or a data URL.</summary>
    public static MediaInput Image(string data)=>Inline("image",data);
    /// <summary>Encode audio bytes; server validates codec and limits.</summary>
    public static MediaInput Audio(ReadOnlySpan<byte> data)=>new("audio",Convert.ToBase64String(data));
    /// <summary>Use audio base64 or a data URL.</summary>
    public static MediaInput Audio(string data)=>Inline("audio",data);
    private static MediaInput Inline(string type,string data)
    {
        Questions.Text(data,"media data");
        if(data.StartsWith("http://",StringComparison.OrdinalIgnoreCase)||data.StartsWith("https://",StringComparison.OrdinalIgnoreCase))throw new ArgumentException("Media requires inline base64 or a data URL");
        return new(type,data);
    }
}
/// <summary>Media token accounting over executed prompts.</summary>
public sealed record MediaUsage(
    [property: JsonPropertyName("images")] long Images,
    [property: JsonPropertyName("audio")] long Audio,
    [property: JsonPropertyName("media_tokens")] long MediaTokens,
    [property: JsonPropertyName("detail")] string Detail);
/// <summary>One ranked tool candidate.</summary>
public sealed record ToolRanking(
    [property: JsonPropertyName("tool")] string Tool,
    [property: JsonPropertyName("probability")] double Probability);
