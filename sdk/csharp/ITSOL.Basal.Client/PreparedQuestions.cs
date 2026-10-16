using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text.Json.Nodes;

namespace ITSOL.Basal;

/// <summary>Typed decisions from reusable, prepared question objects. Import ITSOL.Basal to use this extension.</summary>
public static class PreparedQuestionsExtensions
{
    // An extension keeps existing dictionary/factory instance overloads authoritative.
    // A generic instance overload would capture concrete Dictionary inputs instead.
    /// <summary>Decide using a shape made from QuestionBuilder.Prepare fields. An anonymous object
    /// may select, rename or repeat prepared fields. Each call creates a separate answer shape;
    /// the prepared object never stores responses and may be shared across concurrent decisions.
    /// Named shapes require a public constructor matching all their public answer members; positional records
    /// are supported. Other classes can select their fields into an anonymous object.</summary>
    public static async Task<TypedDecisionResponse<TQuestions>> DecideAsync<
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors | DynamicallyAccessedMemberTypes.PublicProperties | DynamicallyAccessedMemberTypes.PublicFields)] TQuestions>(
        this BasalClient client, JsonNode state, TQuestions questions, DecisionOptions? options = null,
        CancellationToken cancellationToken = default) where TQuestions : notnull
    {
        ArgumentNullException.ThrowIfNull(client);
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(questions);
        var builder = new QuestionBuilder();
        TQuestions answers;
        try { answers = PreparedShape.Copy(questions, builder); }
        finally { builder.Close(); }
        builder.BindShape(answers, nameof(questions));
        var parsed = await client.DecideAsync(state, builder.Questions, options, cancellationToken).ConfigureAwait(false);
        builder.Complete(parsed);
        return new TypedDecisionResponse<TQuestions>(parsed, answers);
    }
}

internal static class PreparedShape
{
    internal static TShape Copy<
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors | DynamicallyAccessedMemberTypes.PublicProperties | DynamicallyAccessedMemberTypes.PublicFields)] TShape>(
        TShape source, QuestionBuilder builder) where TShape : notnull
    {
        if (source.GetType() != typeof(TShape))
            throw new ArgumentException("Use the concrete prepared shape type or select its fields into an anonymous object; base/interface types cannot preserve all questions", "questions");
        var members = new List<MemberInfo>();
        members.AddRange(typeof(TShape).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(property => property.GetMethod?.IsPublic == true && property.GetIndexParameters().Length == 0));
        members.AddRange(typeof(TShape).GetFields(BindingFlags.Public | BindingFlags.Instance));
        var answerMembers = members.Where(member => typeof(IAnswerField).IsAssignableFrom(MemberType(member))).ToArray();
        if (answerMembers.Length == 0)
        {
            if (source is System.Collections.IDictionary)
                throw new ArgumentException("For dynamic sets, use IReadOnlyDictionary<string, QuestionSpec>; for typed shapes, select fields from QuestionBuilder.Prepare", "questions");
            if (members.Any(member => typeof(QuestionSpec).IsAssignableFrom(MemberType(member))))
                throw new ArgumentException("Use QuestionBuilder.Prepare fields such as q.Choice, rather than Questions.Choice definitions, in a typed shape", "questions");
            throw new ArgumentException("The object must contain typed fields created by QuestionBuilder.Prepare", "questions");
        }

        ConstructorInfo? selected = null;
        MemberInfo[]? arguments = null;
        foreach (var constructor in typeof(TShape).GetConstructors(BindingFlags.Public | BindingFlags.Instance))
        {
            var matched = new List<MemberInfo>();
            foreach (var parameter in constructor.GetParameters())
            {
                var member = FindMember(members, parameter);
                if (member is null) break;
                matched.Add(member);
            }
            if (matched.Count != constructor.GetParameters().Length || answerMembers.Any(member => !matched.Contains(member))) continue;
            if (selected is not null)
                throw new ArgumentException("The prepared shape has multiple matching constructors; select its fields into an anonymous object", "questions");
            selected = constructor;
            arguments = matched.ToArray();
        }
        if (selected is null || arguments is null)
            throw new ArgumentException("The prepared shape needs a public constructor matching its answer members; select them into an anonymous object", "questions");

        var values = new object?[arguments.Length];
        for (var index = 0; index < arguments.Length; index++)
        {
            var member = arguments[index];
            var value = ReadMember(member, source);
            if (typeof(IAnswerField).IsAssignableFrom(MemberType(member)))
            {
                if (value is not IAnswerField field)
                    throw new ArgumentException("Prepared question fields must not be null", "questions");
                values[index] = builder.CopyPrepared(field);
            }
            else values[index] = value; // Constructor metadata is shallow-copied.
        }
        TShape copy;
        try { copy = (TShape)selected.Invoke(values); }
        catch (TargetInvocationException error)
        {
            throw new ArgumentException("Could not construct the prepared answer shape", "questions", error.InnerException ?? error);
        }
        for (var index = 0; index < arguments.Length; index++)
        {
            if (typeof(IAnswerField).IsAssignableFrom(MemberType(arguments[index]))
                && !ReferenceEquals(ReadMember(arguments[index], copy), values[index]))
                throw new ArgumentException("The shape constructor must preserve each supplied answer field under its matching member name", "questions");
        }
        return copy;
    }

    private static MemberInfo? FindMember(List<MemberInfo> members, ParameterInfo parameter)
    {
        var exact = members.Where(member => member.Name == parameter.Name && MemberType(member) == parameter.ParameterType).ToArray();
        if (exact.Length == 1) return exact[0];
        if (exact.Length > 1) return null;
        var insensitive = members.Where(member => string.Equals(member.Name, parameter.Name, StringComparison.OrdinalIgnoreCase)
            && MemberType(member) == parameter.ParameterType).ToArray();
        return insensitive.Length == 1 ? insensitive[0] : null;
    }

    private static Type MemberType(MemberInfo member) => member switch
    {
        PropertyInfo property => property.PropertyType,
        FieldInfo field => field.FieldType,
        _ => throw new InvalidOperationException("Only public properties and fields are inspected")
    };

    private static object? ReadMember(MemberInfo member, object source)
    {
        try
        {
            return member switch
            {
                PropertyInfo property => property.GetValue(source),
                FieldInfo field => field.GetValue(source),
                _ => throw new InvalidOperationException("Only public properties and fields are inspected")
            };
        }
        catch (TargetInvocationException error)
        {
            throw new ArgumentException("Could not read a prepared question member", "questions", error.InnerException ?? error);
        }
    }
}
