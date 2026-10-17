// Prompt wording from upstream basal v2.1.0 tasks.py (Apache-2.0).
use crate::prompt::Lang;

pub(super) fn word(lang: Lang, key: &str) -> &'static str {
    match (lang, key) {
        (Lang::Pl, "classify") => "Która kategoria najlepiej opisuje ten tekst?",
        (Lang::Pl, "none") => "Żadna z powyższych kategorii",
        (Lang::Pl, "classify_multi") => "Które kategorie opisują ten tekst?",
        (Lang::Pl, "verify") => "Twierdzenie: {claim}\nCo źródło mówi o tym twierdzeniu?",
        (Lang::Pl, "supported") => "Źródło to potwierdza",
        (Lang::Pl, "contradicted") => "Źródło temu przeczy",
        (Lang::Pl, "not_stated") => "Źródło tego nie mówi",
        (Lang::Pl, "verify_binary") => "Twierdzenie: {claim}\nCzy źródło potwierdza to twierdzenie?",
        (Lang::Pl, "compress") => {
            "Fragment: „{unit}”\nCzy ten fragment zawiera informację potrzebną do odpowiedzi na pytanie: {q}"
        }
        (Lang::Pl, "compress_generic") => "Fragment: „{unit}”\nCzy ten fragment zawiera ważną informację z dokumentu?",
        (Lang::Pl, "line") => "W której linii dokumentu jest wartość pola: {field}?",
        (Lang::Pl, "enum") => "Jaka jest wartość pola: {field}?",
        (Lang::Pl, "span") => "Linia: „{line}”\nKtóry fragment tej linii to dokładnie wartość pola: {field}?",
        (Lang::Pl, "enum_multi") => "Które wartości ma pole: {field}?",
        (Lang::Pl, "missing") => "Nie podano w dokumencie",
        (Lang::Pl, "yes") => "Tak",
        (Lang::Pl, "no") => "Nie",
        (Lang::En, "classify") => "Which label best describes this text?",
        (Lang::En, "none") => "None of the labels above",
        (Lang::En, "classify_multi") => "Which labels describe this text?",
        (Lang::En, "verify") => "Claim: {claim}\nWhat does the source say about this claim?",
        (Lang::En, "supported") => "The source supports it",
        (Lang::En, "contradicted") => "The source contradicts it",
        (Lang::En, "not_stated") => "The source does not say",
        (Lang::En, "verify_binary") => "Claim: {claim}\nDoes the source support this claim?",
        (Lang::En, "compress") => {
            "Passage: \"{unit}\"\nDoes this passage hold information needed to answer this question: {q}"
        }
        (Lang::En, "compress_generic") => {
            "Passage: \"{unit}\"\nDoes this passage hold important information of the document?"
        }
        (Lang::En, "line") => "Which line of the document gives the value of the field: {field}?",
        (Lang::En, "enum") => "What is the value of the field: {field}?",
        (Lang::En, "span") => "Line: \"{line}\"\nWhich part of this line is exactly the value of the field: {field}?",
        (Lang::En, "enum_multi") => "Which values does the field have: {field}?",
        (Lang::En, "missing") => "Not stated in the document",
        (Lang::En, "yes") => "Yes",
        (Lang::En, "no") => "No",
        _ => unreachable!("task prompt keys are static"),
    }
}
