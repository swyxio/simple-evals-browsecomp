import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import jinja2
import numpy as np
import requests
from tqdm import tqdm

from custom_types import EvalResult, Message, SamplerBase, SingleEvalResult

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
]


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<div class="results-grid">
    <div class="result-item">
        <span class="result-label">Correct Answer:</span>
        <span class="result-value">{{ correct_answer }}</span>
    </div>
    <div class="result-item">
        <span class="result-label">Extracted Answer:</span>
        <span class="result-value">{{ extracted_answer }}</span>
    </div>
    <div class="result-item">
        <span class="result-label">Score:</span>
        <span class="result-value">{{ score }}</span>
    </div>
</div>

{% if grader_response %}
<div class="grader-response">
    <h3>Grader Response</h3>
    <div class="grader-response-content">
        <pre>{{ grader_response }}</pre>
    </div>
</div>
{% endif %}

{% if search_data %}
<div class="search-results section">
    <h3 class="section-title">Search Activity</h3>
    {% for search in search_data %}
    <div class="search-query">
        🔍 "{{ search.query }}" 
        <span style="font-size: 0.8em; color: #7f8c8d;">({{ search.status }})</span>
    </div>
    <div class="search-results-content">
        {% if search.results %}
            {% for result in search.results %}
            <div class="search-result">
                <div class="search-result-title">{{ result.title }}</div>
                <div class="search-result-url">{{ result.url }}</div>
            </div>
            {% endfor %}
        {% else %}
            <div class="no-results">No search results available</div>
        {% endif %}
    </div>
    {% endfor %}
</div>
{% endif %}

{% if token_usage %}
<div class="token-usage section">
    <h3 class="section-title">Token Usage</h3>
    <div class="token-usage-content">
        <div class="token-stats">
            <div class="token-stat">
                <span class="token-stat-label">Model:</span>
                <span class="token-stat-value">{{ token_usage.model }}</span>
            </div>
            <div class="token-stat">
                <span class="token-stat-label">Input Tokens:</span>
                <span class="token-stat-value">{{ "{:,}".format(token_usage.input_tokens) }}</span>
            </div>
            <div class="token-stat">
                <span class="token-stat-label">Output Tokens:</span>
                <span class="token-stat-value">{{ "{:,}".format(token_usage.output_tokens) }}</span>
            </div>
            <div class="token-stat">
                <span class="token-stat-label">Total Tokens:</span>
                <span class="token-stat-value">{{ "{:,}".format(token_usage.total_tokens) }}</span>
            </div>
            <div class="token-stat">
                <span class="token-stat-label">Estimated Cost:</span>
                <span class="token-stat-value">${{ "{:.6f}".format(token_usage.estimated_cost) }}</span>
            </div>
        </div>
    </div>
</div>
{% endif %}

<style>
    .section {
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 1.2em;
        color: #2c3e50;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
        margin: 25px 0 15px 0;
    }
    .results-grid {
        display: grid;
        grid-template-columns: max-content 1fr;
        gap: 8px 16px;
        margin: 15px 0;
    }
    .result-item {
        display: contents;
    }
    .result-label {
        font-weight: bold;
        color: #333;
        white-space: nowrap;
    }
    .result-value {
        font-family: monospace;
        word-break: break-word;
    }
    .grader-response, .token-usage, .search-results {
        margin-top: 20px;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .grader-response h3, .token-usage h3, .search-results h3 {
        margin-top: 0;
        color: #333;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }
    .token-usage-content, 
    .grader-response-content,
    .search-results-content {
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
        background-color: #fff;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 3px;
    }
    pre, .token-stats {
        margin: 0;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .token-stats {
        display: grid;
        grid-template-columns: max-content 1fr;
        gap: 5px 15px;
    }
    .token-stat {
        display: contents;
    }
    .token-stat-label {
        font-weight: bold;
    }
    .token-stat-value {
        text-align: right;
    }
    
    .search-query {
        font-weight: bold;
        color: #2c3e50;
        margin: 10px 0 5px 0;
    }
    
    .search-result {
        margin: 8px 0;
        padding: 8px;
        background-color: #f8f9fa;
        border-left: 3px solid #3498db;
    }
    
    .search-result-title {
        font-weight: 500;
        color: #2980b9;
        margin-bottom: 3px;
    }
    
    .search-result-url {
        font-size: 0.85em;
        color: #7f8c8d;
        word-break: break-all;
    }
    
    .no-results {
        color: #7f8c8d;
        font-style: italic;
    }
</style>
"""


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def check_equality(sampler: SamplerBase, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    sampler_response = sampler([dict(content=prompt, role="user")])
    response_text = sampler_response.response_text
    return response_text.lower().strip() == "yes"


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Generate HTML snippet (inside a <div>) for a message.
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Create a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )


def make_report_from_example_htmls(htmls: list[str]):
    """
    Create a standalone HTML report from a list of example htmls
    """
    return jinja_env.from_string(_report_template).render(
        score=None, metrics={}, htmls=htmls
    )


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def url_to_fileobj(url: str, binary=False) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content) if binary else io.StringIO(response.text)


def has_only_user_assistant_messages(messages: list[Message]) -> bool:
    """
    Check if the messages only contain user and assistant messages.
    """
    return all(m["role"] in ("user", "assistant") for m in messages)
