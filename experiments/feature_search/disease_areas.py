"""Arm C: rule-based therapeutic areas from the free-text condition strings.

The snapshot has no MeSH terms, so each condition string is matched against one regular
expression per area (19 areas). A trial gets every area that any of its conditions
matches, plus `no_mapped_area` (no condition matched anything, including trials without
conditions) and `n_areas`.

Started from an earlier pilot mapper. Fixes made after a check of 40 random condition
strings (see README), before any model was fitted with it:
  - "liver" matched "delivery" (gastro); now a whole word
  - "urolog" matched "neurologic" (renal); now word-initial, and "neurolog" maps to neurology
  - "ulcer" sent every ulcer (stasis, pressure, foot) to gastro; gastro now needs
    peptic/gastric/duodenal ulcer, the generic word stays in dermatology
  - added "abscess", "cutaneous" (dermatology) and "alveolar ridge", "edentul" (dental)
and, from reading the patterns, a few substring traps:
  - "burn" matched "heartburn"; "heart" no longer matches "heartburn" (now gastro)
  - "bone" matched "bone marrow"; "vascular" matched "neovascular"; "arter" matched "quarter";
    "mental" matched "developmental"; "ards", "rsv", "sars", "aids" matched inside words;
    "weight" matched "birth weight"; the acute-lymphoblastic abbreviation "ALL" matched the
    English word "all" (now case-sensitive)
A second sample of 40 (seed 7) then showed "retin" matching "transthyretin" (now
word-initial), gastric ulcers also going to dermatology, "hearing aids" going to
infectious; fixed, and "cystitis" (renal) and "obstet" (women's health) added.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

AREAS = {
    "oncology": r"cancer|carcinom|tumou?r|neoplas|oncolog|malignan|lymphoma|leuka?emia|myeloma|melanoma|sarcoma|glioma|glioblastoma|blastoma|mesothelioma|metasta|myelodysplas|myeloproliferative|adenocarcinoma|nsclc|sclc|hodgkin|waldenstr|macroglobulin|mds\b|\baml\b|\bcll\b|(?-i:\bALL\b)|\bcml\b|seminoma|germ cell|astrocytoma|ependymoma|chordoma|carcinoid|neuroendocrine",
    "hematology": r"an(a)?emia|hemophilia|haemophilia|sickle|thalass|thrombocytop|neutropen|coagul|von willebrand|bleeding|hemoglobin|iron deficiency|myelofibrosis|polycythemia|aplastic|\bitp\b|stem cell transplant|graft.versus|gvhd|bone marrow",
    "cardiovascular": r"heart(?!burn)|cardi|coronary|myocard|hypertens|atrial|arrhythm|angina|stroke|cerebrovascular|(?<!neo)vascular|aort|\barter|thrombo|embol|venous|hyperlipid|cholesterol|dyslipid|lipid|infarct|ischemi|ischaemi|valve|peripheral arter|aneurysm",
    "metabolic_endocrine": r"diabet|obes|overweight|insulin|glyc|metabolic|thyroid|adrenal|cushing|acromegal|growth hormone|osteopor|vitamin d|hypogonad|pcos|polycystic ovar|gout|hyperuric|(?<!birth )weight",
    "neurology": r"neurolog|alzheimer|dementia|parkinson|epilep|seizure|multiple sclerosis|migraine|headache|neuropath|amyotrophic|\bals\b|huntington|spinal cord|brain injur|cognitive|neurodegener|ataxia|dystrophy|myasthenia|spinal muscular|cerebral palsy|concussion|neuralg|restless leg|tremor|dystonia",
    "psychiatry": r"depress|schizophren|bipolar|anxiety|psychos|psychiatr|autis|adhd|attention deficit|ptsd|stress disorder|obsessive|insomnia|sleep|eating disorder|anorexia nervosa|bulimia|\bmental|suicid|panic",
    "substance_use": r"smok|tobacco|nicotine|alcohol|opioid use|opiate|cocaine|substance|addict|cannabis|methamphetamine|drug use|dependence",
    "infectious": r"hiv|(?<!hearing )\baids\b|hepatitis|influenza|malaria|tubercul|infect|vaccin|sepsis|bacter|viral|virus|pneumonia|fung|candid|herpes|hpv|papilloma|covid|\bsars\b|cytomegalo|ebola|dengue|parasit|helminth|schistosom|leishman|chagas|cholera|typhoid|meningococc|pneumococc|pertussis|rotavirus|\brsv\b|syncytial|clostridi|staphylo|streptoc|urinary tract infection|otitis",
    "respiratory": r"asthma|copd|pulmonary|lung|bronch|respirat|cystic fibrosis|emphysema|idiopathic pulmonary|rhinitis|sinusitis|apnea|apnoea|\bards\b|cough",
    "immunology_rheum": r"rheumat|arthritis|lupus|psoria|spondyl|crohn|colitis|inflammatory bowel|vasculitis|sjogren|sjögren|scleroderma|sclerosis, systemic|autoimmun|allerg|urticaria|atopic|eczema|dermatitis|celiac|transplant|rejection",
    "musculoskeletal": r"osteoarthr|fracture|back pain|knee|\bhips?\b|shoulder|tendon|ligament|muscle|musculoskeletal|bone(?! marrow)|sarcopenia|orthop|spine|\bdisc\b|joint|rotator",
    "gastro_hepatic": r"\bliver|hepat|cirrho|gastr|bowel|intestin|esophag|oesophag|reflux|gerd|heartburn|pancrea|biliar|colon|constipation|diarrh|irritable|nash|nafld|steatohep|dyspeps|peptic ulcer|duodenal ulcer|hemorrhoid|anal fissure",
    "renal_urology": r"kidney|renal|nephr|dialysis|urinar|bladder|cystitis|incontinen|prostat(?!e cancer)|erectile|overactive|\burolog|hemodialysis",
    "ophthalmology": r"macular|\bretin|glaucoma|cataract|\beyes?\b|ocular|uveitis|myopia|cornea|dry eye|vision|visual|keratitis|conjunctiv",
    "dermatology": r"skin|acne|wound|\bburns?\b|(?<!gastric )(?<!peptic )(?<!duodenal )ulcer|scar|alopecia|vitiligo|rosacea|hidradenitis|keloid|pruritus|onychomycosis|abscess|cutaneous",
    "womens_reproductive": r"obstet|pregnan|preterm|\blabou?r\b|cesarean|caesarean|infertil|ivf|in vitro fert|endometrio|menopaus|contracept|postpartum|preeclampsia|pre-eclampsia|uterine|fibroid|vaginal|ovar|breastfeed|lactation|gestational",
    "pain_anesthesia": r"pain|analges|anesthe|anaesthe|postoperative|nausea|vomiting|sedation|surgery|surgical|perioperative|intubation",
    "oral_dental": r"dental|tooth|teeth|periodont|gingiv|caries|oral mucositis|orthodont|endodont|pulp|alveolar ridge|edentul",
    "healthy": r"healthy|volunteer|normal subjects",
}
_RX = {k: re.compile(v, re.I) for k, v in AREAS.items()}
AREA_COLUMNS = [f"area_{k}" for k in AREAS]
COLUMNS = AREA_COLUMNS + ["no_mapped_area", "n_areas"]


def areas_of(cond: str) -> list[str]:
    hits = [k for k, rx in _RX.items() if rx.search(cond)]
    # oncology dominates incidental organ matches ("lung cancer" is not respiratory here)
    if "oncology" in hits:
        hits = [h for h in hits if h in ("oncology", "hematology", "infectious")]
        if "hematology" in hits and re.search(r"leuk|lymphom|myelom|myelodys", cond, re.I):
            hits.remove("hematology")
    return hits


def area_features(df: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 column per area, `no_mapped_area` and `n_areas`, indexed like df."""
    cache: dict[str, list[str]] = {}
    rows = np.zeros((len(df), len(AREAS)), dtype=float)
    index = {k: i for i, k in enumerate(AREAS)}
    for r, conds in enumerate(df.conditions):
        for c in (list(conds) if conds is not None else []):
            if c not in cache:
                cache[c] = areas_of(c)
            for a in cache[c]:
                rows[r, index[a]] = 1.0
    out = pd.DataFrame(rows, columns=AREA_COLUMNS, index=df.index)
    out["n_areas"] = out[AREA_COLUMNS].sum(axis=1)
    out["no_mapped_area"] = (out.n_areas == 0).astype(float)
    return out[COLUMNS]


def mapper_check(conditions: list[str], n: int = 40, seed: int = 20261119) -> pd.DataFrame:
    """A random sample of distinct condition strings with their areas, for reading by eye."""
    import random
    sample = random.Random(seed).sample(sorted(set(conditions)), n)
    return pd.DataFrame({"condition": sample, "areas": ["; ".join(areas_of(c)) for c in sample]})
