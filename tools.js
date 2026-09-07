// Tool implementations backing the clinical assistant's Claude tool-use loop.
// Each tool calls a public, authoritative medical data source directly over HTTPS.
// No API keys required for any of these.

const FDA_LABEL_URL = 'https://api.fda.gov/drug/label.json';
const PUBMED_ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi';
const PUBMED_ESUMMARY_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi';
const RXNORM_RXCUI_URL = 'https://rxnav.nlm.nih.gov/REST/rxcui.json';
const RXNORM_PROPS_URL = 'https://rxnav.nlm.nih.gov/REST/rxcui';

function truncate(text, max = 600) {
  if (!text) return null;
  const joined = Array.isArray(text) ? text.join(' ') : String(text);
  return joined.length > max ? joined.slice(0, max) + '…' : joined;
}

/** Search FDA drug labeling for a brand or generic drug name. */
async function searchDrugInfo({ query, limit = 3 }) {
  const attempts = [
    `openfda.generic_name:"${query}"`,
    `openfda.brand_name:"${query}"`
  ];

  for (const search of attempts) {
    const url = `${FDA_LABEL_URL}?search=${encodeURIComponent(search)}&limit=${limit}`;
    const res = await fetch(url);
    if (!res.ok) continue;
    const data = await res.json();
    if (!data.results?.length) continue;

    return {
      source: 'FDA drug label database (openFDA)',
      results: data.results.map((r) => ({
        brand_name: r.openfda?.brand_name?.[0] ?? null,
        generic_name: r.openfda?.generic_name?.[0] ?? null,
        manufacturer: r.openfda?.manufacturer_name?.[0] ?? null,
        purpose: truncate(r.purpose),
        indications_and_usage: truncate(r.indications_and_usage),
        dosage_and_administration: truncate(r.dosage_and_administration),
        warnings: truncate(r.warnings ?? r.boxed_warning),
        drug_interactions: truncate(r.drug_interactions)
      }))
    };
  }

  return { source: 'FDA drug label database (openFDA)', results: [], note: 'No matching drug label found.' };
}

/** Search PubMed for medical literature and return titles + links. */
async function searchMedicalLiterature({ query, max_results = 5 }) {
  const searchUrl = `${PUBMED_ESEARCH_URL}?db=pubmed&retmode=json&retmax=${max_results}&term=${encodeURIComponent(query)}`;
  const searchRes = await fetch(searchUrl);
  if (!searchRes.ok) {
    return { source: 'PubMed', results: [], error: `PubMed search failed (${searchRes.status})` };
  }
  const searchData = await searchRes.json();
  const ids = searchData.esearchresult?.idlist ?? [];
  if (!ids.length) {
    return { source: 'PubMed', results: [], note: 'No articles found.' };
  }

  const summaryUrl = `${PUBMED_ESUMMARY_URL}?db=pubmed&retmode=json&id=${ids.join(',')}`;
  const summaryRes = await fetch(summaryUrl);
  if (!summaryRes.ok) {
    return { source: 'PubMed', results: [], error: `PubMed summary fetch failed (${summaryRes.status})` };
  }
  const summaryData = await summaryRes.json();

  const results = ids.map((id) => {
    const item = summaryData.result?.[id];
    if (!item) return null;
    return {
      pmid: id,
      title: item.title,
      authors: (item.authors ?? []).slice(0, 3).map((a) => a.name).join(', '),
      journal: item.source,
      pub_date: item.pubdate,
      url: `https://pubmed.ncbi.nlm.nih.gov/${id}/`
    };
  }).filter(Boolean);

  return { source: 'PubMed (NCBI E-utilities)', results };
}

/** Look up standardized drug nomenclature (RxCUI, synonyms) via RxNorm. */
async function lookupDrugNomenclature({ drug_name }) {
  const rxcuiUrl = `${RXNORM_RXCUI_URL}?name=${encodeURIComponent(drug_name)}`;
  const rxcuiRes = await fetch(rxcuiUrl);
  if (!rxcuiRes.ok) {
    return { source: 'RxNorm', error: `RxNorm lookup failed (${rxcuiRes.status})` };
  }
  const rxcuiData = await rxcuiRes.json();
  const rxcui = rxcuiData.idGroup?.rxnormId?.[0];
  if (!rxcui) {
    return { source: 'RxNorm', note: `No standardized RxNorm entry found for "${drug_name}".` };
  }

  const propsUrl = `${RXNORM_PROPS_URL}/${rxcui}/properties.json`;
  const propsRes = await fetch(propsUrl);
  const propsData = propsRes.ok ? await propsRes.json() : null;

  return {
    source: 'RxNorm (National Library of Medicine)',
    rxcui,
    name: propsData?.properties?.name ?? null,
    synonym: propsData?.properties?.synonym ?? null,
    term_type: propsData?.properties?.tty ?? null
  };
}

export const toolDefinitions = [
  {
    name: 'search_drug_info',
    description: 'Look up FDA-approved drug labeling: purpose, indications, dosage, warnings, and drug interactions for a brand or generic drug name.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Brand name or generic name of the drug.' },
        limit: { type: 'integer', description: 'Max number of matching labels to return (default 3).' }
      },
      required: ['query']
    }
  },
  {
    name: 'search_medical_literature',
    description: 'Search PubMed for peer-reviewed medical literature matching a topic or clinical question. Returns titles, authors, journal, and links.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search terms, e.g. a condition, drug, or clinical question.' },
        max_results: { type: 'integer', description: 'Max number of articles to return (default 5).' }
      },
      required: ['query']
    }
  },
  {
    name: 'lookup_drug_nomenclature',
    description: 'Look up the standardized RxNorm identifier (RxCUI) and canonical name/synonyms for a drug name.',
    input_schema: {
      type: 'object',
      properties: {
        drug_name: { type: 'string', description: 'Drug name to standardize.' }
      },
      required: ['drug_name']
    }
  }
];

export const toolImplementations = {
  search_drug_info: searchDrugInfo,
  search_medical_literature: searchMedicalLiterature,
  lookup_drug_nomenclature: lookupDrugNomenclature
};
