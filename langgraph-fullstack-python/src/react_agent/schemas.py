"""Machine-readable schemas for the multi-agent psychoanalytic analysis.

Phase 5 splits the single analyst into a Supervisor, three school Specialists and
a Synthesizer, so there are four data contracts:

    SupervisorPlan    - the plan the Supervisor writes before any analysis
    SchoolAnalysis    - one specialist's result, each specialist commits to one school
    SynthesisResult   - the combined answer shown to the user
    TheoryInterpretation - one theory-based reading inside a SchoolAnalysis

Phase 7 adds the review contracts:

    DeterministicIssue - a problem found by code, never by a model
    CriticIssue        - one problem the Critic found in the draft synthesis
    CritiqueResult     - the Critic's verdict, its issues and its fix instructions

The schemas describe *text analysis*, never clinical assessment; there is
deliberately no diagnosis, severity, probability or patient-status field. The
critic contracts follow the same rule: no confidence score and no probability,
because an uncalibrated number would look like evidence without being one.

Pydantic validates model output only. The graph state stores
`model_dump(mode="json")` results, so a checkpoint never contains a custom
Pydantic instance.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SpecialistPerspective = Literal[
    "freudian",
    "object_relations",
    "lacanian",
]
"""The school a specialist agent commits to.

Each specialist may only use this label; none of them may answer as another
school. Cross-school comparison belongs to :class:`SynthesisResult`.
"""

EvidenceSchool = Literal[
    "freudian",
    "object_relations",
    "lacanian",
    "general",
]
"""Shelf a retrieved chunk came from. `general` holds shared notes."""


def _clean_evidence_ids(values: list[str]) -> list[str]:
    """Trim, drop empty entries and de-duplicate evidence ids, keeping order.

    Args:
        values: Raw ids as returned by the model.

    Returns:
        Cleaned ids; ids the model wrapped in punctuation are repaired.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip().strip(",;")
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _normalize_escaped_text(value: str) -> str:
    r"""Replace JSON-style escape sequences that DeepSeek leaves in its text.

    When the model answers through the function-calling channel, it sometimes
    emits ``\n`` and ``\"`` inside the string values instead of real characters.
    Those sequences would reach the chat message as visible backslashes, so they
    are rewritten here: line breaks become real line breaks and escaped quotes
    become ordinary quotes. Only these sequences are touched.

    Args:
        value: Text produced by the model.

    Returns:
        The same text with literal escape sequences replaced.
    """
    if "\\" not in value:
        return value
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


class EvidenceItem(BaseModel):
    """One retrieved theory passage from the local knowledge base.

    Evidence is produced by the retriever, never by a language model: a specialist
    may only *reference* these items. `text` is the real chunk stored in the
    vector database, and the metadata is whatever the loader actually read - a
    missing page or year stays `None` instead of being guessed.

    This is theory material, not a statement about the user, and not a clinical
    document.
    """

    evidence_id: str = Field(
        description="Stable chunk id of the passage, e.g. 'freud_dreams_000123'.",
    )
    school: EvidenceSchool = Field(
        description="School shelf the passage belongs to; 'general' is shared.",
    )
    text: str = Field(
        description="The retrieved passage itself, copied verbatim from the index.",
    )
    source_id: str = Field(
        description="Identifier of the source document the passage belongs to.",
    )
    title: str | None = Field(
        default=None,
        description="Document title, when the knowledge file provides one.",
    )
    author: str | None = Field(
        default=None, description="Author, when the knowledge file states one."
    )
    work_title: str | None = Field(
        default=None,
        description="Title of the work the passage belongs to, when known.",
    )
    year: int | None = Field(
        default=None,
        description="Publication year, only when the knowledge file states one.",
    )
    page: int | None = Field(
        default=None,
        description=(
            "Real page number, only when the loader could read one. Never "
            "invented for markdown or plain text files."
        ),
    )
    section: str | None = Field(
        default=None,
        description="Section heading, when the source file has one.",
    )
    source_path: str | None = Field(
        default=None,
        description="Path of the source file relative to the knowledge root.",
    )
    retrieval_score: float | None = Field(
        default=None,
        description="Cosine similarity reported by the retriever; None if unknown.",
    )


class TheoryInterpretation(BaseModel):
    """One theory-based reading of the material, with its own uncertainty note.

    A theoretical association is never a fact about the user. It must stay in
    this structure and must carry an explicit statement of why it is uncertain.
    """

    perspective: SpecialistPerspective = Field(
        description=(
            "The school this reading belongs to. It must match the specialist "
            "that is answering: a Freudian specialist only uses 'freudian', "
            "and so on. Do not label a reading as another school."
        ),
    )
    claim: str = Field(
        description=(
            "The interpretation itself, phrased as a possibility "
            "(for example 'one reading is ...', 'this may be understood as ...'). "
            "Never phrased as a fact about the user's life, history or diagnosis."
        ),
    )
    textual_basis: list[str] = Field(
        description=(
            "Short quotations or paraphrases of material the user actually "
            "provided that support this reading. Quote or closely restate the "
            "user's own words only. Never include invented background, invented "
            "plot details, or details the user did not supply."
        ),
    )
    uncertainty: str = Field(
        description=(
            "Why this reading is uncertain: which information is missing, what "
            "alternative readings remain open, and how strongly the textual "
            "basis constrains the claim. A bare 'maybe' is not sufficient."
        ),
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of the retrieved theory passages (EvidenceItem.evidence_id) "
            "that support this reading. Only ids supplied in the prompt may be "
            "used - never invent one. Use an empty list when no theory passage "
            "was provided or none applies, and state in `uncertainty` that the "
            "reading is not backed by local literature."
        ),
    )

    @field_validator("claim", "uncertainty")
    @classmethod
    def _normalize_claim_text(cls, value: str) -> str:
        return _normalize_escaped_text(value)

    @field_validator("textual_basis")
    @classmethod
    def _normalize_basis_text(cls, value: list[str]) -> list[str]:
        return [_normalize_escaped_text(item) for item in value]

    @field_validator("evidence_ids")
    @classmethod
    def _normalize_evidence_ids(cls, value: list[str]) -> list[str]:
        return _clean_evidence_ids(value)


class SupervisorPlan(BaseModel):
    """The Supervisor's plan for one analysis turn.

    The Supervisor plans and never analyses. It decides what each specialist
    should pay attention to, whether the user asked for a clinical judgement,
    and what the Synthesizer has to resolve.
    """

    task_summary: str = Field(
        description=(
            "Neutral restatement of what the user is asking for in this turn, "
            "based only on the conversation. No interpretation and no detail "
            "the user did not provide."
        ),
    )
    analysis_focus: str = Field(
        description=(
            "What this analysis most needs to attend to, such as a repeated "
            "narrative feature, an ambiguity, or missing context. Describe the "
            "material, not the user's personality or history."
        ),
    )
    freudian_focus: str = Field(
        description=(
            "What the Freudian specialist should examine: which textual "
            "features matter for unconscious conflict, defence, wish or "
            "symbolism. This is analytical direction only - never a factual "
            "claim about the user (for example never 'the user has childhood "
            "trauma')."
        ),
    )
    object_relations_focus: str = Field(
        description=(
            "What the object-relations specialist should examine: which "
            "textual features matter for internal objects, splitting, "
            "idealisation, projection or relationship patterns. Analytical "
            "direction only, never invented family history."
        ),
    )
    lacanian_focus: str = Field(
        description=(
            "What the Lacanian specialist should examine: which textual "
            "features matter for desire, lack, signifiers and the relation "
            "between subject and language. Analytical direction only."
        ),
    )
    clinical_diagnosis_requested: bool = Field(
        default=False,
        description=(
            "True when the user asks for a clinical judgement, such as whether "
            "they have a disorder. The whole pipeline then declines diagnosis."
        ),
    )
    synthesis_goal: str = Field(
        description=(
            "What the Synthesizer has to resolve: which disagreements between "
            "the three schools matter here, and what the final answer should "
            "help the user see. Not an answer by itself."
        ),
    )

    @field_validator("task_summary", "analysis_focus", "synthesis_goal")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return _normalize_escaped_text(value)

    @field_validator(
        "freudian_focus",
        "object_relations_focus",
        "lacanian_focus",
    )
    @classmethod
    def _normalize_focus(cls, value: str) -> str:
        return _normalize_escaped_text(value)


class SchoolAnalysis(BaseModel):
    """One specialist's structured result.

    The specialist may only work inside its own school. Anything that the user
    did not provide belongs in `interpretations`, never in `observations`.
    """

    perspective: SpecialistPerspective = Field(
        description=(
            "The school that produced this analysis. It must be the school the "
            "specialist was assigned, and must match every entry in "
            "`interpretations`."
        ),
    )
    observations: list[str] = Field(
        description=(
            "Plain statements of what the user explicitly provided: their "
            "words, images, narrative gaps and relationship patterns. Add no "
            "emotions, memories, family details, childhood events or traits "
            "that were not written. For a one-line input such as a dream about "
            "water, the observation is only that - nothing about mothers, "
            "oceans, fear or trauma."
        ),
    )
    interpretations: list[TheoryInterpretation] = Field(
        description=(
            "Readings in this specialist's own school. May be empty when the "
            "material is too thin: saying there is not enough to interpret is "
            "better than inventing an association, and repetition of generic "
            "school vocabulary is not analysis."
        ),
    )
    limitations: list[str] = Field(
        description=(
            "What this school's reading cannot support here: missing context, "
            "ambiguity, the gap between text and the user's life, and the "
            "absence of verifiable sources. Short inputs need at least one."
        ),
    )
    questions: list[str] = Field(
        description=(
            "At most three questions that the user could actually answer and "
            "that would change this school's reading. No symptom checklists."
        ),
    )
    summary: str = Field(
        description=(
            "The specialist's short conclusion in its own voice, consistent "
            "with the fields above. A few sentences, not a full essay."
        ),
    )
    clinical_diagnosis_refused: bool = Field(
        default=False,
        description=(
            "True when the user asked for a clinical judgement and this "
            "specialist therefore declines to diagnose and stays at the level "
            "of theory. Never confirm or exclude a disorder."
        ),
    )

    @field_validator("observations", "limitations", "questions", mode="after")
    @classmethod
    def _normalize_list_text(cls, value: list[str]) -> list[str]:
        return [_normalize_escaped_text(item) for item in value]

    @field_validator("summary")
    @classmethod
    def _normalize_summary(cls, value: str) -> str:
        return _normalize_escaped_text(value)


class SynthesisResult(BaseModel):
    """The combined result that the user finally reads.

    The Synthesizer compares the three specialist results instead of analysing
    the material from scratch. It must keep the disagreement visible rather
    than elect one school as correct.
    """

    common_ground: list[str] = Field(
        description=(
            "Points that at least two schools support on the same textual "
            "basis, stated as readings rather than facts."
        ),
    )
    differences: list[str] = Field(
        description=(
            "Where the schools genuinely diverge: what each one emphasises, "
            "what it explains less well, and why they disagree. Name the "
            "schools explicitly."
        ),
    )
    integrated_interpretation: str = Field(
        description=(
            "The combined reading: what can be said once the three schools are "
            "compared, including what remains undecided between them. Do not "
            "pick a winner and do not invent material."
        ),
    )
    limitations: list[str] = Field(
        description=(
            "Limits of the combined interpretation: missing context, theory "
            "versus the user's real life, and the absence of verifiable "
            "primary sources."
        ),
    )
    follow_up_questions: list[str] = Field(
        description=(
            "At most three questions that would move the analysis forward, "
            "phrased so the user can answer them. No diagnostic screening."
        ),
    )
    clinical_diagnosis_refused: bool = Field(
        default=False,
        description=(
            "True when the user asked for a clinical judgement. The final "
            "answer must then say clearly that this system cannot diagnose."
        ),
    )
    used_evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of the theory passages the synthesis actually relied on, "
            "chosen only from the evidence given to the specialists. The "
            "system renders the source list from these ids, so an id that was "
            "not supplied must never appear here."
        ),
    )
    final_response: str = Field(
        description=(
            "The complete natural-language answer shown to the user. Natural, "
            "specific and readable in Chinese: it must not be a concatenation "
            "of the three summaries, and must not contain JSON, field names, "
            "schema names or tool-call syntax."
        ),
    )

    @field_validator(
        "common_ground",
        "differences",
        "limitations",
        "follow_up_questions",
        mode="after",
    )
    @classmethod
    def _normalize_list_text(cls, value: list[str]) -> list[str]:
        return [_normalize_escaped_text(item) for item in value]

    @field_validator("integrated_interpretation", "final_response")
    @classmethod
    def _normalize_synthesis_text(cls, value: str) -> str:
        return _normalize_escaped_text(value)

    @field_validator("used_evidence_ids")
    @classmethod
    def _normalize_used_evidence_ids(cls, value: list[str]) -> list[str]:
        return _clean_evidence_ids(value)


DeterministicIssueCategory = Literal[
    "invalid_evidence_reference",
    "perspective_mismatch",
    "clinical_safety_mismatch",
    "citation_rendering_error",
    "state_consistency",
]
"""What kind of problem the code-only validator found."""

DeterministicSeverity = Literal["warning", "error"]


class DeterministicIssue(BaseModel):
    """One problem found by code, before any model is asked to review.

    These issues are never produced by a language model: they are facts about
    the state (an id that does not exist, a flag that contradicts the plan, an
    answer that already leaked into the chat history). Anything that can be
    decided by reading data must be decided here instead of by the Critic.
    """

    category: DeterministicIssueCategory = Field(
        description="Which deterministic check failed.",
    )
    message: str = Field(
        description="What exactly is wrong, in one reviewable sentence.",
    )
    related_agent: str | None = Field(
        default=None,
        description="Node whose output is affected, for example 'synthesizer'.",
    )
    related_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence ids involved in the problem, when there are any.",
    )
    severity: DeterministicSeverity = Field(
        default="error",
        description=(
            "`error` means the draft must not be shown as it is; `warning` "
            "records a deviation that the Critic should still look at."
        ),
    )

    @field_validator("message")
    @classmethod
    def _normalize_message(cls, value: str) -> str:
        return _normalize_escaped_text(value)

    @field_validator("related_evidence_ids")
    @classmethod
    def _normalize_related_ids(cls, value: list[str]) -> list[str]:
        return _clean_evidence_ids(value)


CriticIssueCategory = Literal[
    "observation_fidelity",
    "evidence_support",
    "theoretical_consistency",
    "overinterpretation",
    "clinical_safety",
    "synthesis_quality",
]
"""The six things the Critic is allowed to complain about."""

CriticPerspective = Literal[
    "freudian",
    "object_relations",
    "lacanian",
    "integrative",
]
"""Which part of the draft an issue belongs to; `integrative` is the synthesis."""

CriticSeverity = Literal["warning", "error"]


class CriticIssue(BaseModel):
    """One problem the Critic found in the draft synthesis.

    The Critic reviews reasoning, not evidence ids: a citation can exist and
    still be used for a claim it does not support. Every issue must carry an
    actionable instruction, because the revision step may only fix what the
    Critic named.
    """

    category: CriticIssueCategory = Field(
        description="Which review dimension this issue belongs to.",
    )
    severity: CriticSeverity = Field(
        description=(
            "`error` means the draft must be revised; `warning` is a note that "
            "does not by itself require a revision."
        ),
    )
    description: str = Field(
        description=(
            "What is wrong, concretely: name the claim or sentence and say why "
            "it is not supported by the material or the evidence."
        ),
    )
    affected_claim: str | None = Field(
        default=None,
        description="The draft sentence or claim this issue is about, if any.",
    )
    related_perspective: CriticPerspective | None = Field(
        default=None,
        description="Which school or the synthesis itself the issue belongs to.",
    )
    related_evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of retrieved passages this issue refers to. Only ids that were "
            "actually retrieved this turn may appear; the graph validates them "
            "against the same whitelist the specialists use."
        ),
    )
    revision_instruction: str = Field(
        description=(
            "What the revision must do about it: soften, attribute to the "
            "school, delete, or state the missing information. Never an "
            "instruction to add new material."
        ),
    )

    @field_validator("description", "revision_instruction")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return _normalize_escaped_text(value)

    @field_validator("related_evidence_ids")
    @classmethod
    def _normalize_related_ids(cls, value: list[str]) -> list[str]:
        return _clean_evidence_ids(value)


class CritiqueResult(BaseModel):
    """The Critic's verdict on the draft synthesis.

    Deliberately free of scores: there is no confidence value and no
    probability, because an uncalibrated number would look like a measurement
    without being one. The verdict is a decision, and the flags say which
    dimensions were checked.
    """

    verdict: Literal["pass", "revise"] = Field(
        description=(
            "`pass` when the draft is grounded, inside the theory boundaries "
            "and safe to show. `revise` only for problems that affect "
            "correctness, grounding, boundaries or safety - never for style or "
            "for wanting more detail."
        ),
    )
    issues: list[CriticIssue] = Field(
        default_factory=list,
        description="Every problem found; empty is normal for a `pass`.",
    )
    summary: str = Field(
        description="One short paragraph: what was checked and what was found.",
    )
    revision_instructions: list[str] = Field(
        default_factory=list,
        description=(
            "The concrete changes a revision must make, in priority order. "
            "Empty when the verdict is `pass`."
        ),
    )
    clinical_safety_ok: bool = Field(
        description=(
            "False when the draft diagnoses, implies a disorder or draws a "
            "clinical conclusion from thin material."
        ),
    )
    evidence_grounding_ok: bool = Field(
        description=(
            "False when a claim is not supported by the passage it cites, or "
            "when retrieved theory is presented as a fact about the user."
        ),
    )
    observation_fidelity_ok: bool = Field(
        description=(
            "False when the draft treats something the user never wrote as an "
            "observation (childhood, family, trauma, traits, emotions)."
        ),
    )

    @field_validator("summary")
    @classmethod
    def _normalize_summary(cls, value: str) -> str:
        return _normalize_escaped_text(value)

    @field_validator("revision_instructions", mode="after")
    @classmethod
    def _normalize_instructions(cls, value: list[str]) -> list[str]:
        return [_normalize_escaped_text(item) for item in value]
