// Manual live SDK/model comparison, not a CI test framework.
using ITSOL.Basal;
using System.Text.Json;
using System.Text.Json.Nodes;
using static ITSOL.Basal.Questions;

var rootIndex=Array.IndexOf(args,"--measurement-artifacts");
if(rootIndex<0||rootIndex+1>=args.Length)throw new ArgumentException("--measurement-artifacts required");
var root=args[rootIndex+1];
using var client=new BasalClient(Environment.GetEnvironmentVariable("BASAL_URL")??"http://127.0.0.1:19439",timeout:TimeSpan.FromMinutes(5));
Emit(new {sdk="csharp-map",models=await client.ModelsAsync()});
var diagnostic=new Dictionary<string,QuestionSpec>{{"n",Noul("Is this a diagnostic?")}};
try {await client.DecideAsync(JsonValue.Create("diagnostic")!,diagnostic,new DecisionOptions(Model:"__sdk_unknown_model__"));throw new InvalidOperationException("Unknown model accepted");}
catch(BasalHttpException error) when(error.StatusCode==422){Emit(new {sdk="csharp-map",check="unknown-model",status=error.StatusCode});}
bool rejected=false;
try {await client.DecideAsync(JsonValue.Create("diagnostic")!,diagnostic,new DecisionOptions(Media:Array.Empty<MediaInput>()));}
catch(ArgumentException){rejected=true;}
if(!rejected)throw new InvalidOperationException("Empty media accepted");
Emit(new {sdk="csharp-map",check="empty-media",status="local-rejection"});
foreach(var line in File.ReadLines(Path.Combine(root,"inputs/cases.jsonl"))) {
    var c=JsonNode.Parse(line)!.AsObject();
    if(c["media"] is not null&&Environment.GetEnvironmentVariable("BASAL_MEDIA")!="1")continue;
    var definitions=c["questions"]!.AsObject().ToDictionary(p=>p.Key,p=>Definition(p.Value!.AsObject()));
    IReadOnlyList<MediaInput>? media=null;
    if(c["media"] is JsonArray inputs)media=inputs.Select(v=>{
        var m=v!.AsObject();var bytes=File.ReadAllBytes(Path.Combine(root,TextValue(m,"file")));
        var type=TextValue(m,"type");var encoding=TextValue(m,"encoding");
        if(encoding=="bytes")return type=="image"?Media.Image(bytes):Media.Audio(bytes);
        var data=Convert.ToBase64String(bytes);
        if(encoding=="data-url")data=$"data:{TextValue(m,"mime")};base64,{data}";
        return type=="image"?Media.Image(data):Media.Audio(data);
    }).ToArray();
    var options=new DecisionOptions(Model:Environment.GetEnvironmentVariable("BASAL_MODEL"),Lang:c["lang"]?.GetValue<string>(),Media:media,Detail:c["detail"]?.GetValue<string>());
    var response=await client.DecideAsync(c["state"]!.DeepClone(),definitions,options);
    Emit(new {sdk="csharp-map",@case=TextValue(c,"id"),answers=response.Answers.ToDictionary(p=>p.Key,p=>Project(p.Value)),raw=response.Raw});
    if(TextValue(c,"id")=="named-api") {
        var route=(ToolQuestion)definitions["route"];var labels=(MultiQuestion)definitions["labels"];
        var inline=await client.DecideAsync(c["state"]!.DeepClone(),q=>new {route=q.Tool(route),labels=q.Multi(labels)},options);
        Emit(new {sdk="csharp-inline",@case="named-api",answers=new {route=Project(inline.Answers.route.Result),labels=Project(inline.Answers.labels.Result)},raw=inline.Raw});
        var explicitFields=await client.DecideAsync(c["state"]!.DeepClone(),q=>new {route=q.Add("route",route),labels=q.Add("labels",labels)},options);
        Emit(new {sdk="csharp-explicit",@case="named-api",answers=new {route=Project(explicitFields.Answers.route.Result),labels=Project(explicitFields.Answers.labels.Result)},raw=explicitFields.Raw});
        var prepared=QuestionBuilder.Prepare(q=>new {route=q.Tool(route),labels=q.Multi(labels)});
        var reused=await client.DecideAsync(c["state"]!.DeepClone(),prepared,options);
        Emit(new {sdk="csharp-prepared",@case="named-api",answers=new {route=Project(reused.Answers.route.Result),labels=Project(reused.Answers.labels.Result)},raw=reused.Raw});
    }
}
static void Emit(object value)=>Console.WriteLine(JsonSerializer.Serialize(value));
static string TextValue(JsonObject o,string key)=>o[key]!.GetValue<string>();
static Dictionary<string,string> Criteria(JsonObject o)=>o["criteria"]!.Deserialize<Dictionary<string,string>>()!;
static MultiOptions Selection(JsonObject q)=>new(q["threshold"]?.GetValue<double>(),q["min"]?.GetValue<int>(),q["max"]?.GetValue<int>(),q["objective"]?.GetValue<string>()??"threshold",q["exclusive"]?.Deserialize<string[]>(),q["fallback"]?.GetValue<string>(),q["fallback_threshold"]?.GetValue<double>());
static QuestionSpec Definition(JsonObject q) {
    var instruction=TextValue(q,"instructions");
    return TextValue(q,"type") switch {
        "choice"=>Choice(instruction,Criteria(q)),
        "noul"=>Noul(instruction),
        "score"=>Score(instruction,q["criteria"]!.Deserialize<string[]>()!),
        "multi"=>Multi(instruction,Criteria(q),Selection(q)),
        "act"=>Act(instruction,q["costs"]!.Deserialize<Dictionary<string,Dictionary<string,double>>>()!,new ActOptions(q["criteria"]?.Deserialize<Dictionary<string,string>>(),DeferAction:q["defer_action"]?.GetValue<string>(),MaxError:q["max_error"]?.GetValue<double>())),
        "tool"=>Tool(instruction,q["tools"]!.AsArray().Select(v=>v!.AsObject()).ToArray(),new ToolOptions(q["extra"]?.Deserialize<string[]>(),q["rank"]?.GetValue<int>(),q["multi"]?.GetValue<bool>()??false,q["max_error"]?.GetValue<double>(),Selection(q))),
        _=>throw new ArgumentException("Unsupported diagnostic question")
    };
}
static object Project(AnswerResult a)=>a switch {
    ChoiceResult v=>new {type="choice",choice=v.Choice,confidence=v.Confidence,probabilities=v.Probabilities},
    NoulResult v=>new {type="noul",noul=v.Noul,confidence=v.Confidence,probabilities=v.Probabilities},
    ScoreResult v=>new {type="score",score=v.Score,confidence=v.Confidence,legend=v.Legend,probabilities=v.Probabilities},
    MultiResult v=>SelectionProjection(v),
    ActResult v=>new {type="act",action=v.Action,answer=v.Outcome,expected_costs=v.ExpectedCosts,confidence=v.Confidence,calibration=v.Calibration,refused=v.Refused,max_error=v.MaxError,deferred=v.Deferred,probabilities=v.Probabilities},
    ToolResult v=>new {type="tool",tool=v.Tool,call=v.Call,confidence=v.Confidence,multi=v.Multi,tools=v.Tools,ranking=v.Ranking,selection=v.Selection is null?null:SelectionProjection(v.Selection),deferred=v.Deferred,refused=v.Refused,max_error=v.MaxError,probabilities=v.Probabilities},
    _=>throw new ArgumentException("Unsupported answer")
};
static object SelectionProjection(MultiResult v)=>new {type="multi",selected=v.Selected,threshold=v.Threshold,set_confidence=v.SetConfidence,objective=v.Objective,expected_score=v.ExpectedScore,constraints_applied=v.ConstraintsApplied,probabilities=v.Probabilities};
