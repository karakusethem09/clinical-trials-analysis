"""Remove statements that a trial itself stopped, before any text model reads it.

The registry keeps only the current text, and some sponsors edit the title or summary
after a trial stops ("This study was terminated due to slow accrual"; "planned 450
patients, only 28 were randomised"). Those sentences would let a text model read the
outcome. The rule removes a sentence when it

  1. has the study, trial, protocol, enrolment, accrual, recruitment, sponsor or project
     as the subject of a past-tense stopping verb (terminated, withdrawn, halted,
     stopped, closed, suspended, discontinued, cancelled, ended), with no modal verb
     ("may", "will", "if", "once") in between or just before, so protocol stopping
     rules are kept;
  2. has a stopping verb with the study or trial as its object ("the sponsor
     terminated this study"), unless participants are the subject ("patients who
     discontinued the study"), or names the termination, closure or suspension of
     the study or trial;
  3. reports poor accrual ("slow accrual", "low enrollment", "lack of funding"); or
  4. compares the planned and the actual number of participants ("only 28 randomized",
     "28 of the planned 450").

"Study" followed by drug, treatment, visit and similar words is not a subject, so
"study drug was discontinued" and "may stop the growth of tumour cells" are kept.
Only the title and the brief summary are masked: eligibility criteria describe who
may join, and the stopping words in them are about participants ("withdrawal of
life-sustaining treatment", "discontinue NSAIDs"), not about the study.
"""
from __future__ import annotations

import re

SUBJECT = (r"\b(study|trial|protocol|enrol(l)?ment|accrual|recruitment|sponsor|project)\b"
           r"(?!\s+(drugs?|medications?|treatments?|interventions?|products?|agents?|vaccines?|devices?|"
           r"visits?|periods?|procedures?|participation|sites?|staff|team|doctors?|physicians?|"
           r"population|subjects?|participants?|patients?|arms?|groups?|days?|weeks?|therapy|dose|"
           r"medicine|materials?|questionnaires?|assessments?|nurses?|coordinators?|objectives?|design|aims?|"
           r"results?|data|completion|endpoints?))")
# "closed loop", "closed-suction drain", "closed reduction" use the word as an adjective
VERB = (r"\b(terminated|withdrawn|withdrew|halted|stopped|closed|suspended|discontinued|"
        r"cancell?ed|ended)\b(?!-|\s+(loop|reduction|suction|system|head|fracture|chest|circuit|"
        r"wound|incision|technique|eyes?|kinetic|cohort|form))")
MODAL = r"\b(may|might|will|would|can|could|should|shall|must|if|unless|once|until|whether|going)\b"

STUDY_STOPPED = re.compile(rf"{SUBJECT}(?P<gap>(\W+\w+){{0,6}}?)\W+{VERB}", re.IGNORECASE)
STOPPED_THE_STUDY = re.compile(
    r"\b(terminated|halted|stopped|closed|suspended|discontinued|cancell?ed|ended)\s+"
    r"(the|this|its|their)\s+(study|trial|enrol(l)?ment|recruitment|accrual)\b"
    r"(?!\s+(drugs?|medications?|treatments?|interventions?|products?|visits?|participation))",
    re.IGNORECASE)
PARTICIPANT = re.compile(r"\b(subjects?|participants?|patients?|volunteers?|people|persons?|women|men|"
                         r"children|infants?|who|those)\b", re.IGNORECASE)
STOPPING_OF_STUDY = re.compile(
    r"\b(termination|closure|suspension|discontinuation|cancell?ation|early stopping)\s+of\s+"
    r"(the|this)\s+(study|trial)\b", re.IGNORECASE)
POOR_ACCRUAL = re.compile(
    r"\b((slow|low|poor|insufficient|inadequate|limited)\s+(patient\s+|participant\s+|subject\s+)?"
    r"(accrual|enrol\w*|recruit\w*)|lack\s+of\s+(funding|accrual|enrol\w*|recruit\w*)|"
    r"(study|trial)\s+(was\s+|has\s+been\s+)?never\s+(started|opened|initiated))\b", re.IGNORECASE)
PLANNED_VS_ACTUAL = re.compile(
    r"\bplanned\b[^.]{0,80}?\bonly\s+\d+|"
    r"\bonly\s+\d+\s+(\w+\s+){0,2}(were\s+|was\s+|have\s+been\s+|had\s+been\s+)?"
    r"(enrol\w*|randomi[sz]ed|recruited|included|treated)\b|"
    r"\b\d+\s+(of|out\s+of)\s+(the\s+)?(planned|target(ed)?|intended|anticipated|expected)\s+\d+",
    re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")

# The rule used before this one, which masked any sentence with a stopping word. Kept
# only to report how much the narrower rule changes (reports/masking_summary.json).
BROAD_RULE = re.compile(
    r"\b(terminat\w*|withdr[ae]wn?|withdraw\w*|halt\w*|stopp\w*|stop|discontinu\w*|"
    r"suspen\w*|cancel\w*|abandon\w*|prematur\w*|closed|closure|close[sd]? early|"
    r"futility|(slow|low|poor|insufficient|inadequate) (accrual|enrol\w*|recruit\w*)|"
    r"lack of (funding|accrual|enrol\w*|recruit\w*)|did not (start|enrol\w*|recruit\w*))\b",
    re.IGNORECASE)


def states_stopping(sentence: str) -> bool:
    """True if the sentence says the study itself stopped or under-enrolled."""
    for m in STUDY_STOPPED.finditer(sentence):
        before = " ".join(sentence[:m.start()].split()[-4:])
        if not re.search(MODAL, m.group("gap") + " " + before, re.IGNORECASE):
            return True
    for m in STOPPED_THE_STUDY.finditer(sentence):
        before = " ".join(sentence[:m.start()].split()[-4:])
        if not (PARTICIPANT.search(before) or re.search(MODAL, before, re.IGNORECASE)):
            return True
    return bool(STOPPING_OF_STUDY.search(sentence) or POOR_ACCRUAL.search(sentence)
                or PLANNED_VS_ACTUAL.search(sentence))


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE.split(text or "") if s.strip()]


def mask_stopping(text: str | None) -> tuple[str, list[str]]:
    """Drop sentences that state the study stopped. Returns the kept text and the dropped sentences."""
    kept, dropped = [], []
    for sentence in split_sentences(text or ""):
        (dropped if states_stopping(sentence) else kept).append(sentence)
    return " ".join(kept), dropped


def document(title: str | None, conditions: str | None, summary: str | None,
             criteria: str | None) -> tuple[str, list[str]]:
    """The text the model reads: masked title and summary, conditions and criteria as they are."""
    title_kept, title_dropped = mask_stopping(title)
    summary_kept, summary_dropped = mask_stopping(summary)
    parts = [title_kept, conditions or "", summary_kept, criteria or ""]
    return " \n ".join(p for p in parts if p), title_dropped + summary_dropped
