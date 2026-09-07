# Knowledge corpus (RAG)

Every file here is chunked, embedded, and made citable by the agent — it checks
this corpus first on every query and treats local policy as authoritative.

## Layout

```
knowledge/
  guidelines/        specialty-society & national practice guidelines (ACC/AHA, IDSA, ADA, GOLD, KDIGO, NICE, WHO…)
  protocols/         your institution's order sets, stewardship policy, care pathways, escalation criteria
  drug-references/   FDA labels (SPL), package inserts, renal/hepatic dose-adjustment tables
  reviews/           Cochrane, USPSTF, AHRQ evidence reports
  local/             where your institution diverges from national guidance, and why
```

Supported file types: `.pdf` `.md` `.markdown` `.txt` `.html` `.htm`.

## Provenance (required for auditable citations)

Each document needs **source, date, version, jurisdiction**. Two ways to supply it:

**1. Sidecar** — a `<name>.meta.json` next to any file:

```json
{
  "source": "Infectious Diseases Society of America",
  "title": "Management of Community-Acquired Pneumonia in Adults",
  "published": "2019-10",
  "revised": "2023-06",
  "version": "2019 (2023 update)",
  "jurisdiction": "US",
  "url": "https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/"
}
```

**2. Front-matter** — for `.md`/`.txt`, a block at the very top:

```markdown
---
source: KDIGO
title: Clinical Practice Guideline for Diabetes Management in CKD
published: 2022
version: "2022"
jurisdiction: international
url: https://kdigo.org/guidelines/diabetes-ckd/
---

# Recommendation 1.1
...
```

Retrieved passages are then cited as
`Local KB · <title> › <heading>  (Source · version · date · jurisdiction)`.

## Adding a document

```powershell
.\.venv\Scripts\python.exe -m api.kb_load "C:\downloads\idsa-cap-2019.pdf" `
  --category guidelines `
  --source "IDSA/ATS" --title "Community-Acquired Pneumonia in Adults" `
  --published 2019-10 --version 2019 --jurisdiction US `
  --url "https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/"
```

This copies the file into the right folder, writes the sidecar, and reindexes.
Or drop files in by hand (with sidecars) and run `python -m api.rag` /
`POST /api/rag/reindex`.

## Where to get content

- **Open / public-domain:** USPSTF recommendations, CDC, AHRQ, many WHO guidelines
  (CC BY-NC-SA), NICE syndication (free API key).
- **Society guidelines:** most publish free full text (journal supplements / society
  sites) — download the PDF and load it. Respect each publisher's licence for
  redistribution; keeping the corpus local to your institution is normal.
- **Drug labels:** DailyMed (`dailymed.nlm.nih.gov`) has downloadable SPL for every
  approved product — public domain.

## Notes

- `.kb_index.json` is generated here and is git-ignored.
- `README.md` and `*.meta.json` files are not themselves indexed.
- The `SAMPLE-*.md` file is a placeholder template — replace it.
