import re
from dataclasses import asdict, dataclass

from app.router.tokens import estimate_text_tokens
from app.schemas.chat import ChatCompletionRequest, ImagePart, Message, TextPart

# Only the most recent text is scanned for patterns, so a huge pasted document costs the same
# as a modest one. (Token counts still cover everything: they are a cheap, linear pass.)
_SCAN_LIMIT = 20_000
_SMALL_TALK_MAX_TOKENS = 12

# --- patterns -----------------------------------------------------------------------------
# Kept simple and anchored: no nested quantifiers, so no input can make them backtrack badly.

_CODE_KEYWORD = re.compile(
    r"^\s*(?:def|class|import|from|return|function|const|let|var|public|private|static|"
    r"if|elif|else|for|while|try|catch|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\b",
    re.IGNORECASE,
)
_CODE_LINE = re.compile(
    r"=>|;\s*$|^\s*[{}]\s*$|</?[a-z][a-z0-9]*[^>\n]{0,80}>|\w\(.{0,60}\)\s*[{:]\s*$"
)
_FENCE = "```"

_MATH_SYMBOLS = re.compile(r"[∫∑∏√∂∇≈≠≤≥±∞πθλμσΔ]")
# Inline maths is $...$ with no space just inside either dollar sign and no digit straight after
# the closing one, which is what keeps "$5 for lunch and $7 for dinner" from reading as a formula.
_MATH_LATEX = re.compile(
    r"\$[^\s$](?:[^$\n]{0,198}[^\s$])?\$(?!\d)"
    r"|\\(?:frac|sum|int|sqrt|lim|infty|alpha|beta|theta|partial|begin)\b"
)
_MATH_EQUATION = re.compile(
    r"\d\s*[-+*/^]\s*\d+\s*=|\b[a-z]\s*=\s*[-+]?\d+\s*[-+*/^]", re.IGNORECASE
)
_MATH_WORDS = re.compile(
    r"\b(?:integral|derivative|theorem|lemma|proof|eigenvalue|eigenvector|matrix|determinant|"
    r"polynomial|logarithm|calculus|differential equation|probability distribution|"
    r"standard deviation|combinatorics|modular arithmetic)\b",
    re.IGNORECASE,
)

_REASONING_CUES = re.compile(
    r"\b(?:step[ -]by[ -]step|explain why|why does|why is|how does|analy[sz]e|analysis|"
    r"compare|contrast|trade-?offs?|pros and cons|evaluate|critique|justify|derive|prove|"
    r"design|architect(?:ure)?|optimi[sz]e|debug|refactor|root cause|reason about|"
    r"implications?|strategy|algorithm|time complexity|edge cases?|invariants?|"
    r"first principles|rigorous(?:ly)?|in depth|comprehensive)\b",
    re.IGNORECASE,
)

_CONSTRAINT_CUES = re.compile(
    # Longer phrases first: in an alternation the first match wins, so "must" would hide "must not".
    r"\b(?:must not|should not|do not|don't|no more than|no fewer than|ensure that|at least|"
    r"at most|must|never|exactly|only|without|constraints?|requirements?)\b",
    re.IGNORECASE,
)

_STRUCTURED_OUTPUT = re.compile(
    r"\b(?:respond|reply|answer|return|output|format|write|give)\b[^.\n]{0,40}"
    r"\b(?:json|yaml|xml|csv|table|schema)\b",
    re.IGNORECASE,
)

_LIST_ITEM = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+\S", re.MULTILINE)

_SMALL_TALK = re.compile(
    r"^\W*(?:hi|hello|hey|yo|thanks|thank you|thx|ok|okay|yes|no|sure|great|cool|bye|"
    r"good (?:morning|afternoon|evening|night)|how are you|what'?s up)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ComplexityFeatures:
    """Signals about how demanding a request is. Raw measurements; scoring happens elsewhere."""

    prompt_tokens: int  # the whole conversation, estimated
    last_user_tokens: int  # the final user turn: what is actually being asked
    system_tokens: int
    turns: int  # user messages so far: how deep the conversation is
    code_blocks: int  # fenced ``` blocks
    code_line_ratio: float  # share of non-empty lines that look like code, 0 to 1
    math_signals: int  # equations, LaTeX, symbols and maths vocabulary
    reasoning_cues: int  # distinct phrases like "step by step", "trade-offs", "prove"
    constraint_cues: int  # distinct phrases like "must", "exactly", "do not"
    question_marks: int  # in the final user turn
    list_items: int  # numbered or bulleted items in the final user turn: multi-part asks
    structured_output: bool  # asks for JSON, YAML, a table, a schema ...
    tool_count: int  # tools offered
    forces_tool: bool  # the caller insists on a tool call
    image_count: int
    requested_completion_tokens: int | None  # max_tokens, if set: how long an answer is wanted
    non_ascii_ratio: float  # multilingual text is harder for small models
    small_talk: bool  # a greeting, thanks, or a one-word reply

    @property
    def has_code(self) -> bool:
        return self.code_blocks > 0 or self.code_line_ratio >= 0.3

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _text_of(message: Message) -> str:
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


def _recent_text(texts: list[str]) -> str:
    """The latest text, newest last, trimmed to at most ``_SCAN_LIMIT`` characters."""
    chunks: list[str] = []
    remaining = _SCAN_LIMIT
    for text in reversed(texts):
        if remaining <= 0:
            break
        chunks.append(text[-remaining:])
        remaining -= len(chunks[-1])
    return "\n".join(reversed(chunks))


def _code_stats(text: str) -> tuple[int, float]:
    """(fenced blocks, share of non-empty lines that look like code)."""
    blocks = text.count(_FENCE) // 2 + (1 if text.count(_FENCE) % 2 else 0)
    in_fence = False
    code_lines = total_lines = 0
    for line in text.splitlines():
        if line.strip().startswith(_FENCE):
            in_fence = not in_fence
            continue
        if not line.strip():
            continue
        total_lines += 1
        if in_fence or _CODE_KEYWORD.search(line) or _CODE_LINE.search(line):
            code_lines += 1
    return blocks, (code_lines / total_lines if total_lines else 0.0)


def _math_signals(text: str) -> int:
    return (
        len(_MATH_SYMBOLS.findall(text))
        + len(_MATH_LATEX.findall(text))
        + len(_MATH_EQUATION.findall(text))
        + len(_MATH_WORDS.findall(text))
    )


def _distinct(pattern: re.Pattern[str], text: str) -> int:
    return len({match.lower() for match in pattern.findall(text)})


def extract_features(request: ChatCompletionRequest) -> ComplexityFeatures:
    """Measure a request. Pure, deterministic and fast whatever the size of the prompt."""
    messages = request.messages
    texts = [_text_of(m) for m in messages]
    last_user = next((_text_of(m) for m in reversed(messages) if m.role == "user"), "")
    recent = _recent_text(texts)
    scanned_last_user = last_user[-_SCAN_LIMIT:]

    blocks, code_ratio = _code_stats(recent)
    non_ascii = sum(1 for ch in recent if ord(ch) > 127)
    last_user_tokens = estimate_text_tokens(last_user)
    named_tool = request.tool_choice is not None and request.tool_choice not in ("none", "auto")

    return ComplexityFeatures(
        prompt_tokens=sum(estimate_text_tokens(t) for t in texts),
        last_user_tokens=last_user_tokens,
        system_tokens=sum(
            estimate_text_tokens(t)
            for m, t in zip(messages, texts, strict=True)
            if m.role == "system"
        ),
        turns=sum(1 for m in messages if m.role == "user"),
        code_blocks=blocks,
        code_line_ratio=round(code_ratio, 4),
        math_signals=_math_signals(recent),
        reasoning_cues=_distinct(_REASONING_CUES, scanned_last_user),
        constraint_cues=_distinct(_CONSTRAINT_CUES, scanned_last_user),
        question_marks=scanned_last_user.count("?"),
        list_items=len(_LIST_ITEM.findall(scanned_last_user)),
        structured_output=bool(_STRUCTURED_OUTPUT.search(scanned_last_user)),
        tool_count=len(request.tools or []),
        forces_tool=named_tool,
        image_count=sum(
            1
            for m in messages
            if isinstance(m.content, list)
            for part in m.content
            if isinstance(part, ImagePart)
        ),
        requested_completion_tokens=request.max_tokens,
        non_ascii_ratio=round(non_ascii / len(recent), 4) if recent else 0.0,
        small_talk=last_user_tokens <= _SMALL_TALK_MAX_TOKENS
        and bool(_SMALL_TALK.search(scanned_last_user.strip())),
    )
