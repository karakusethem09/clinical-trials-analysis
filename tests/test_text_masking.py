from text_masking import document, mask_stopping, states_stopping

import pytest


@pytest.mark.parametrize("sentence", [
    "This study was terminated due to slow accrual.",
    "The trial closed early.",
    "Novartis voluntarily terminated this study.",
    "Enrollment was stopped at 2253 patients due to low recruitment.",
    "Termination of the study was decided by the sponsor.",
    "We planned 450 patients but only 28 were randomized.",
    "28 of the planned 450 participants enrolled.",
    "Due to limited enrollment, this study was early terminated.",
    "The sponsor has halted the trial",
])
def test_statements_about_the_study_stopping_are_masked(sentence: str) -> None:
    assert states_stopping(sentence)


@pytest.mark.parametrize("sentence", [
    "Discontinue NSAIDs 7 days before surgery.",
    "The drug may stop the growth of tumour cells.",
    "Withdrawal of life-sustaining treatment is not planned.",
    "Study drug was discontinued in 3 patients.",
    "The study may be stopped if safety concerns arise.",
    "Participants may withdraw from the study at any time.",
    "Patients who discontinued the study early are replaced.",
    "Randomized Trial Comparing Sitagliptin to Placebo in Closed Loop",
    "Once the study has ended, participants will be offered the extension study.",
])
def test_innocent_sentences_are_kept(sentence: str) -> None:
    assert not states_stopping(sentence)


def test_mask_returns_kept_text_and_dropped_sentences() -> None:
    kept, dropped = mask_stopping("A phase 2 study of drug X. The study was terminated early. Patients get X daily.")
    assert dropped == ["The study was terminated early."]
    assert kept == "A phase 2 study of drug X. Patients get X daily."


def test_document_masks_title_and_summary_but_not_criteria() -> None:
    text, dropped = document("Study of X (terminated)", "Asthma",
                             "This trial was halted by the sponsor. It tests X.",
                             "Subjects withdrawn from a study that was terminated.")
    assert dropped == ["Study of X (terminated)", "This trial was halted by the sponsor."]
    assert "It tests X." in text and "Subjects withdrawn from a study that was terminated." in text
