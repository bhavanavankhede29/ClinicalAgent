"""Adapter interface: a de-identified EHR stay -> CaseRecord.

No data ships with this repo. You obtain the datasets under their own access
regime, then point an adapter at your local copy:

  * MIMIC-IV / MIMIC-IV-ED / eICU-CRD  — PhysioNet credentialing + CITI training
    (https://physionet.org/content/mimiciv/) then a signed data use agreement.
  * Institutional EHR                  — IRB approval + de-identification
    (HIPAA Safe Harbor or Expert Determination) before it reaches this code.
  * Claims (CMS, MarketScan) / SEER    — their respective DUAs.

`CaseRecord` deliberately has no birth date, no admission date, and no free-text
notes — only an age band, coded problems/meds/labs, the orders actually placed,
and outcomes. `to_case_record()` enforces that.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .schemas import CaseRecord

_AGE_BANDS = [(0, 18, "0-17"), (18, 45, "18-44"), (45, 65, "45-64"),
              (65, 75, "65-74"), (75, 85, "75-84"), (85, 200, "85+")]


def age_band(age: float | int | None) -> str | None:
    if age is None:
        return None
    a = min(float(age), 89.0)  # cap per Safe Harbor
    for lo, hi, label in _AGE_BANDS:
        if lo <= a < hi:
            return label
    return "85+"


def to_case_record(
    *, case_id: str, dataset: str, age: float | int | None = None, sex: str | None = None,
    problems: list[str] | None = None, medications: list[str] | None = None,
    labs: dict[str, str] | None = None, procedures: list[str] | None = None,
    orders: list[str] | None = None, outcomes: dict[str, str] | None = None,
) -> CaseRecord:
    """Build a CaseRecord, stripping anything identifying."""
    return CaseRecord(
        case_id=str(case_id), dataset=dataset,
        age_group=age_band(age), sex=(sex or "").lower()[:1] or None,
        problems=sorted(set(problems or [])),
        medications=sorted(set(medications or [])),
        labs={k: str(v) for k, v in (labs or {}).items()},
        procedures=sorted(set(procedures or [])),
        orders=list(orders or []),
        outcomes={k: str(v) for k, v in (outcomes or {}).items()},
    )


# --- MIMIC-IV skeleton ------------------------------------------------------
# Fill these in against your local extract. Kept minimal on purpose: the join
# logic and column names depend on which MIMIC tables/version you pulled.

def mimic_iv_stays(root: str | Path) -> Iterator[CaseRecord]:
    """Yield one CaseRecord per hospital admission from a MIMIC-IV CSV extract.

    Expected files under `root` (adjust to your extract):
        hosp/admissions.csv  hosp/diagnoses_icd.csv  hosp/prescriptions.csv
        hosp/labevents.csv   hosp/procedures_icd.csv  hosp/patients.csv
    """
    root = Path(root)
    adm = root / "hosp" / "admissions.csv"
    if not adm.exists():
        raise FileNotFoundError(
            f"{adm} not found. Download MIMIC-IV from PhysioNet after credentialing "
            "and point --root at the extracted folder.")

    # --- Sketch only. Real use needs per-hadm_id joins across the tables above.
    with adm.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            yield to_case_record(
                case_id=row.get("hadm_id", ""),
                dataset="mimic-iv",
                age=None,                         # derive from patients.anchor_age
                problems=[], medications=[], labs={}, procedures=[], orders=[],
                outcomes={
                    "died_in_hospital": row.get("hospital_expire_flag", ""),
                    "admission_type": row.get("admission_type", ""),
                },
            )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preview CaseRecords from a MIMIC-IV extract.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=3)
    a = ap.parse_args()
    for i, cr in enumerate(mimic_iv_stays(a.root)):
        if i >= a.limit:
            break
        print(cr.model_dump_json(indent=2, exclude_none=True))
