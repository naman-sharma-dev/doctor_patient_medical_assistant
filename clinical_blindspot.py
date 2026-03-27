"""
clinical_blindspot.py
=====================
A module for the Doctor-Patient Medical Assistant that helps clinicians
identify commonly *overlooked* (blind-spot) conditions based on a patient's
reported symptoms, demographics, and medical history.

Usage example
-------------
    from clinical_blindspot import ClinicalBlindspotDetector, PatientProfile

    patient = PatientProfile(
        age=45,
        sex="female",
        symptoms=["fatigue", "weight gain", "cold intolerance", "dry skin"],
        medical_history=["hypertension"],
    )

    detector = ClinicalBlindspotDetector()
    results = detector.analyze(patient)
    for alert in results:
        print(alert)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Set


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatientProfile:
    """Demographic and clinical snapshot of a patient."""

    age: int
    sex: str                           # "male" | "female" | "other"
    symptoms: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sex = self.sex.lower().strip()
        self.symptoms = [s.lower().strip() for s in self.symptoms]
        self.medical_history = [h.lower().strip() for h in self.medical_history]
        self.current_medications = [m.lower().strip() for m in self.current_medications]
        self.family_history = [f.lower().strip() for f in self.family_history]


@dataclass
class BlindspotAlert:
    """
    Represents a single flagged blind-spot condition for a patient.

    Attributes
    ----------
    condition : str
        Name of the commonly missed condition.
    matched_symptoms : list[str]
        Symptoms from the patient profile that matched this condition.
    match_score : float
        Fraction of the condition's key symptoms present in the patient profile
        (0.0 – 1.0).
    priority : str
        "high" | "medium" | "low" — derived from match_score and severity.
    description : str
        Brief explanation of why this condition is a common clinical blind spot.
    recommended_tests : list[str]
        Suggested diagnostic investigations to rule in / rule out the condition.
    """

    condition: str
    matched_symptoms: List[str]
    match_score: float
    priority: str
    description: str
    recommended_tests: List[str]

    def __str__(self) -> str:
        lines = [
            f"[{self.priority.upper()}] Possible blind-spot: {self.condition}",
            f"  Match score      : {self.match_score:.0%}",
            f"  Matched symptoms : {', '.join(self.matched_symptoms) or 'none'}",
            f"  Why it's missed  : {self.description}",
            f"  Recommended tests: {', '.join(self.recommended_tests)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Blind-spot knowledge base
# ---------------------------------------------------------------------------

# Each entry describes one commonly missed condition.
# Fields:
#   key_symptoms       – symptoms that strongly suggest the condition
#   supporting_symptoms– additional symptoms that can add confidence
#   risk_ages          – (min_age, max_age) tuple; None means any age
#   risk_sex           – set of sexes at elevated risk; None means any
#   risk_history       – medical-history / family-history items that raise risk
#   severity           – "high" | "medium" | "low" (affects priority weighting)
#   description        – why this condition is a clinical blind spot
#   recommended_tests  – first-line investigations to order

BLINDSPOT_DATABASE: List[Dict] = [
    {
        "condition": "Hypothyroidism",
        "key_symptoms": {"fatigue", "weight gain", "cold intolerance", "dry skin",
                         "constipation", "hair loss", "depression", "bradycardia"},
        "supporting_symptoms": {"muscle weakness", "joint pain", "heavy menstrual periods",
                                "slowed heart rate", "puffy face"},
        "risk_ages": (30, 80),
        "risk_sex": {"female"},
        "risk_history": {"autoimmune disease", "thyroid disease", "family history of thyroid"},
        "severity": "medium",
        "description": (
            "Hypothyroidism is frequently attributed to ageing, stress, or depression, "
            "causing it to be under-diagnosed—particularly in women and older adults."
        ),
        "recommended_tests": ["TSH", "Free T4", "Thyroid antibodies (TPO-Ab, TgAb)"],
    },
    {
        "condition": "Type 2 Diabetes (undiagnosed)",
        "key_symptoms": {"increased thirst", "frequent urination", "fatigue",
                         "blurred vision", "slow-healing wounds", "recurrent infections"},
        "supporting_symptoms": {"unexplained weight loss", "tingling hands or feet",
                                "darkened skin patches"},
        "risk_ages": (35, 90),
        "risk_sex": None,
        "risk_history": {"obesity", "hypertension", "family history of diabetes",
                         "gestational diabetes", "prediabetes"},
        "severity": "high",
        "description": (
            "Up to one-third of people with type 2 diabetes are undiagnosed because early "
            "symptoms are non-specific and often normalised."
        ),
        "recommended_tests": ["Fasting glucose", "HbA1c", "Oral glucose tolerance test (OGTT)"],
    },
    {
        "condition": "Celiac Disease",
        "key_symptoms": {"bloating", "diarrhea", "abdominal pain", "fatigue",
                         "weight loss", "anemia"},
        "supporting_symptoms": {"mouth ulcers", "itchy skin rash", "joint pain",
                                "depression", "infertility"},
        "risk_ages": None,
        "risk_sex": None,
        "risk_history": {"autoimmune disease", "irritable bowel syndrome",
                         "family history of celiac"},
        "severity": "medium",
        "description": (
            "Celiac disease is commonly misdiagnosed as IBS or non-specific GI disorders; "
            "average time to diagnosis can exceed a decade."
        ),
        "recommended_tests": ["tTG-IgA antibody", "Total IgA", "HLA-DQ2/DQ8 genotyping",
                              "Duodenal biopsy"],
    },
    {
        "condition": "Sleep Apnea",
        "key_symptoms": {"snoring", "daytime sleepiness", "morning headaches",
                         "poor concentration", "fatigue"},
        "supporting_symptoms": {"witnessed apnea", "nocturia", "mood changes",
                                "high blood pressure"},
        "risk_ages": (30, 80),
        "risk_sex": None,
        "risk_history": {"obesity", "hypertension", "cardiovascular disease"},
        "severity": "high",
        "description": (
            "Sleep apnea—especially in women—is frequently overlooked because presentations "
            "differ from the classic male pattern, and patients may not report snoring."
        ),
        "recommended_tests": ["Epworth Sleepiness Scale", "Polysomnography (PSG)",
                              "Home sleep apnea test (HSAT)", "Pulse oximetry"],
    },
    {
        "condition": "Depression / Major Depressive Disorder",
        "key_symptoms": {"persistent sadness", "loss of interest", "fatigue",
                         "sleep disturbances", "poor concentration", "appetite changes"},
        "supporting_symptoms": {"weight changes", "psychomotor agitation",
                                "feelings of worthlessness", "recurrent thoughts of death"},
        "risk_ages": None,
        "risk_sex": None,
        "risk_history": {"anxiety", "previous depressive episode", "chronic pain",
                         "substance abuse", "family history of depression"},
        "severity": "high",
        "description": (
            "Depression often presents with somatic complaints (fatigue, pain, GI symptoms) "
            "rather than low mood, causing primary care clinicians to miss it."
        ),
        "recommended_tests": ["PHQ-9 screening", "Beck Depression Inventory",
                              "Thyroid function (to rule out hypothyroidism)",
                              "CBC, metabolic panel"],
    },
    {
        "condition": "Atrial Fibrillation (paroxysmal)",
        "key_symptoms": {"palpitations", "fatigue", "shortness of breath",
                         "dizziness", "chest discomfort"},
        "supporting_symptoms": {"exercise intolerance", "near-syncope",
                                "reduced exercise tolerance"},
        "risk_ages": (55, 90),
        "risk_sex": None,
        "risk_history": {"hypertension", "heart failure", "coronary artery disease",
                         "thyroid disease", "family history of AFib"},
        "severity": "high",
        "description": (
            "Paroxysmal AFib is easily missed on a standard ECG because the arrhythmia may "
            "not be present at the time of recording, leading to delayed anticoagulation."
        ),
        "recommended_tests": ["12-lead ECG", "Holter monitor (24–48 h)",
                              "Echocardiogram", "TSH"],
    },
    {
        "condition": "Vitamin B12 Deficiency",
        "key_symptoms": {"fatigue", "tingling hands or feet", "weakness",
                         "poor memory", "depression", "mouth ulcers"},
        "supporting_symptoms": {"pale skin", "glossitis", "balance problems",
                                "vision disturbances"},
        "risk_ages": (50, 90),
        "risk_sex": None,
        "risk_history": {"vegetarian or vegan diet", "gastrectomy", "metformin use",
                         "proton pump inhibitor use", "pernicious anemia"},
        "severity": "medium",
        "description": (
            "B12 deficiency is frequently attributed to ageing or stress; neurological "
            "damage can be irreversible if treatment is delayed."
        ),
        "recommended_tests": ["Serum B12", "Methylmalonic acid (MMA)", "Homocysteine",
                              "CBC with differential", "Peripheral blood smear"],
    },
    {
        "condition": "Pulmonary Embolism",
        "key_symptoms": {"sudden shortness of breath", "chest pain", "rapid heart rate",
                         "coughing up blood", "leg pain or swelling"},
        "supporting_symptoms": {"low-grade fever", "sweating", "light-headedness",
                                "anxiety"},
        "risk_ages": (30, 90),
        "risk_sex": None,
        "risk_history": {"recent surgery", "prolonged immobility", "cancer",
                         "previous deep vein thrombosis", "oral contraceptive use",
                         "pregnancy"},
        "severity": "high",
        "description": (
            "PE can mimic musculoskeletal chest pain, pneumonia, or anxiety and is one of "
            "the most frequently missed diagnoses in emergency settings."
        ),
        "recommended_tests": ["Wells PE score", "D-dimer", "CT pulmonary angiography (CTPA)",
                              "ABG", "ECG", "Chest X-ray"],
    },
    {
        "condition": "Polycystic Ovary Syndrome (PCOS)",
        "key_symptoms": {"irregular periods", "excess hair growth", "acne",
                         "weight gain", "thinning scalp hair"},
        "supporting_symptoms": {"skin darkening", "fatigue", "mood changes",
                                "difficulty conceiving"},
        "risk_ages": (15, 45),
        "risk_sex": {"female"},
        "risk_history": {"insulin resistance", "family history of PCOS",
                         "type 2 diabetes"},
        "severity": "medium",
        "description": (
            "PCOS is under-recognised because it presents heterogeneously and symptoms are "
            "often normalised or attributed to lifestyle factors."
        ),
        "recommended_tests": ["Pelvic ultrasound", "LH:FSH ratio", "Free and total testosterone",
                              "DHEA-S", "Fasting insulin and glucose", "AMH"],
    },
    {
        "condition": "Obstructive Lung Disease (COPD / Asthma)",
        "key_symptoms": {"chronic cough", "shortness of breath", "wheezing",
                         "increased mucus production", "chest tightness"},
        "supporting_symptoms": {"recurrent respiratory infections", "fatigue",
                                "exercise intolerance", "cyanosis"},
        "risk_ages": None,
        "risk_sex": None,
        "risk_history": {"smoking", "occupational dust exposure", "asthma",
                         "family history of COPD", "alpha-1 antitrypsin deficiency"},
        "severity": "high",
        "description": (
            "COPD is frequently under-diagnosed because breathlessness is attributed to "
            "deconditioning or age, and spirometry is underutilised in primary care."
        ),
        "recommended_tests": ["Spirometry (pre- and post-bronchodilator)", "Chest X-ray",
                              "CT chest", "Alpha-1 antitrypsin level"],
    },
]


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class ClinicalBlindspotDetector:
    """
    Analyses a ``PatientProfile`` against a knowledge base of commonly missed
    conditions and returns a prioritised list of ``BlindspotAlert`` objects.

    Parameters
    ----------
    min_match_score : float
        Minimum fraction of key symptoms that must be present before a
        condition is flagged (default: 0.30, i.e. 30 %).
    """

    PRIORITY_THRESHOLDS: Dict[str, float] = {
        "high":   0.60,
        "medium": 0.40,
        "low":    0.00,
    }

    def __init__(self, min_match_score: float = 0.30) -> None:
        self.min_match_score = min_match_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, patient: PatientProfile) -> List[BlindspotAlert]:
        """
        Evaluate a patient profile against the blindspot knowledge base.

        Returns
        -------
        list[BlindspotAlert]
            Alerts sorted by priority (high → medium → low) then by match score
            (descending).  Empty list when no blind spots are flagged.
        """
        alerts: List[BlindspotAlert] = []

        patient_symptoms: Set[str] = set(patient.symptoms)
        patient_history: Set[str] = set(patient.medical_history + patient.family_history)

        for entry in BLINDSPOT_DATABASE:
            # --- Age filter ---
            if entry["risk_ages"] is not None:
                min_age, max_age = entry["risk_ages"]
                if not (min_age <= patient.age <= max_age):
                    continue

            # --- Sex filter (None means any sex qualifies) ---
            if entry["risk_sex"] is not None:
                if patient.sex not in entry["risk_sex"] and patient.sex != "other":
                    continue

            # --- Symptom matching ---
            key_symptoms: Set[str] = entry["key_symptoms"]
            matched = patient_symptoms & key_symptoms
            match_score = len(matched) / len(key_symptoms) if key_symptoms else 0.0

            # Boost score slightly when supporting symptoms are present
            supporting: Set[str] = entry.get("supporting_symptoms", set())
            supporting_matched = patient_symptoms & supporting
            if supporting and supporting_matched:
                boost = 0.05 * len(supporting_matched)
                match_score = min(1.0, match_score + boost)

            if match_score < self.min_match_score:
                continue

            # --- History boost ---
            risk_history: Set[str] = entry.get("risk_history", set())
            history_overlap = patient_history & risk_history
            if history_overlap:
                match_score = min(1.0, match_score + 0.10)

            # --- Priority determination ---
            severity = entry.get("severity", "medium")
            priority = self._compute_priority(match_score, severity)

            alert = BlindspotAlert(
                condition=entry["condition"],
                matched_symptoms=sorted(matched | supporting_matched),
                match_score=match_score,
                priority=priority,
                description=entry["description"],
                recommended_tests=entry["recommended_tests"],
            )
            alerts.append(alert)

        return self._sort_alerts(alerts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_priority(self, match_score: float, severity: str) -> str:
        """Map match score + clinical severity to a priority label."""
        # Start from base score; high-severity conditions get a 15 % bump
        effective_score = match_score + (0.15 if severity == "high" else 0.0)

        if effective_score >= self.PRIORITY_THRESHOLDS["high"]:
            return "high"
        if effective_score >= self.PRIORITY_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    @staticmethod
    def _sort_alerts(alerts: List[BlindspotAlert]) -> List[BlindspotAlert]:
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            alerts,
            key=lambda a: (priority_order[a.priority], -a.match_score),
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_blindspots(
    age: int,
    sex: str,
    symptoms: List[str],
    medical_history: List[str] | None = None,
    family_history: List[str] | None = None,
    current_medications: List[str] | None = None,
    min_match_score: float = 0.30,
) -> List[BlindspotAlert]:
    """
    One-call convenience wrapper around ``ClinicalBlindspotDetector``.

    Parameters
    ----------
    age : int
        Patient age in years.
    sex : str
        "male", "female", or "other".
    symptoms : list[str]
        Reported symptoms.
    medical_history : list[str], optional
        Past and current diagnoses.
    family_history : list[str], optional
        Relevant family-history items.
    current_medications : list[str], optional
        Medications the patient is currently taking.
    min_match_score : float
        Minimum symptom match fraction to flag a condition (default 0.30).

    Returns
    -------
    list[BlindspotAlert]
        Prioritised list of flagged conditions.
    """
    patient = PatientProfile(
        age=age,
        sex=sex,
        symptoms=symptoms,
        medical_history=medical_history or [],
        family_history=family_history or [],
        current_medications=current_medications or [],
    )
    detector = ClinicalBlindspotDetector(min_match_score=min_match_score)
    return detector.analyze(patient)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Print a demonstration analysis to stdout."""

    print("=" * 60)
    print("Clinical Blind-Spot Detector — Demo")
    print("=" * 60)

    cases = [
        {
            "label": "Case 1: Middle-aged woman with fatigue & weight gain",
            "age": 45,
            "sex": "female",
            "symptoms": ["fatigue", "weight gain", "cold intolerance", "dry skin",
                         "depression", "constipation"],
            "medical_history": ["hypertension"],
            "family_history": ["family history of thyroid"],
        },
        {
            "label": "Case 2: Obese 55-year-old male with snoring & daytime sleepiness",
            "age": 55,
            "sex": "male",
            "symptoms": ["snoring", "daytime sleepiness", "morning headaches",
                         "fatigue", "poor concentration"],
            "medical_history": ["obesity", "hypertension"],
            "family_history": [],
        },
        {
            "label": "Case 3: Young woman with GI symptoms & anemia",
            "age": 28,
            "sex": "female",
            "symptoms": ["bloating", "diarrhea", "abdominal pain", "fatigue",
                         "anemia", "mouth ulcers"],
            "medical_history": ["irritable bowel syndrome"],
            "family_history": ["family history of celiac"],
        },
    ]

    for case in cases:
        print(f"\n{case['label']}")
        print("-" * 60)
        alerts = detect_blindspots(
            age=case["age"],
            sex=case["sex"],
            symptoms=case["symptoms"],
            medical_history=case.get("medical_history"),
            family_history=case.get("family_history"),
        )
        if alerts:
            for alert in alerts:
                print(alert)
                print()
        else:
            print("No blind spots flagged.")

    print("=" * 60)
    print("DISCLAIMER: This tool is for clinical decision support only.")
    print("It does not replace professional medical judgement.")
    print("=" * 60)


if __name__ == "__main__":
    _demo()
