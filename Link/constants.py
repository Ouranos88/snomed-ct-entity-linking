SNOMED_FSN_TYPE = "900000000000003001"   # Fully Specified Name (includes semantic tag)
SNOMED_SYN_TYPE = "900000000000013009"   # Synonym (clean display term)

CONTEXT_WINDOW = 300  # chars of surrounding text shown to LLM for reranking

# Expand French ophthalmological abbreviations before FAISS queries only —
# the original verbatim span is preserved for display and offset purposes.
FR_OPHTH_ABBREVS = {
    "CA":    "chambre antérieure",
    "OD":    "oeil droit",
    "OG":    "oeil gauche",
    "ODG":   "les deux yeux",
    "AV":    "acuité visuelle",
    "PIO":   "pression intraoculaire",
    "LAF":   "lampe à fente",
    "FO":    "fond d'oeil",
    "DR":    "décollement de rétine",
    "DMLA":  "dégénérescence maculaire liée à l'âge",
    "GCA":   "glaucome chronique à angle ouvert",
    "IOL":   "implant intraoculaire",
    "DMEK":  "kératoplastie endothéliale membrane de Descemet",
    "DSAEK": "kératoplastie lamellaire postérieure automatisée",
    "OCT":   "tomographie par cohérence optique",
    "HTO":   "hypertonie oculaire",
    "HTA":   "hypertension artérielle",
    "RGO":   "reflux gastro-oesophagien",
    "IV":    "intraveineux",
    "SC":    "sous-cutané",
}


def expand_abbrevs(text: str) -> str:
    """Word-by-word expansion: "CA profonde" → "chambre antérieure profonde"."""
    return " ".join(FR_OPHTH_ABBREVS.get(w, w) for w in text.split())
