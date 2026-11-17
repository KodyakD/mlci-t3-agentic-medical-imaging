#!/usr/bin/env python3
"""
Synthetic dataset of triage reports for the Kaggle Chest X-Ray (Paul Mooney) project.

Each row = one fake case:
  INPUT  : CNN outputs (class, probabilities, Grad-CAM ratio) + free-text clinical note
  OUTPUT : the report text (same style as your static template)

Usage:
    python generate_reports_dataset.py --n 6000 --seed 42
Produces: reports_dataset.jsonl, reports_dataset.csv
"""
import argparse
import csv
import json
import random

# ----------------------------------------------------------------------------
# Symptoms: (canonical key, [free-text variants], pneumonia-compatible?)
# ----------------------------------------------------------------------------
SYMPTOMS = {
    "fievre":         (["fievre", "fievre a 39", "fievre depuis 3 jours", "temperature elevee", "febrile"], True),
    "toux":           (["toux", "toux seche", "toux productive", "toux grasse depuis 5 jours", "toux persistante"], True),
    "dyspnee":        (["dyspnee", "essoufflement", "difficulte a respirer", "souffle court", "gene respiratoire"], True),
    "douleur_thoracique": (["douleur thoracique", "douleur au thorax", "point de cote", "douleur en respirant"], True),
    "expectorations": (["expectorations purulentes", "crachats jaunes", "expectorations", "crachats verdatres"], True),
    "frissons":       (["frissons", "frissons et sueurs", "sueurs nocturnes"], True),
    "fatigue":        (["fatigue", "asthenie", "grande fatigue", "malaise general"], True),
    "tachypnee":      (["tachypnee", "respiration rapide", "polypnee"], True),
    "cyanose":        (["cyanose", "levres bleutees", "desaturation"], True),
    # non-specific / not suggestive of pneumonia
    "cephalees":      (["maux de tete", "cephalees"], False),
    "nausees":        (["nausees", "douleurs abdominales", "diarrhee"], False),
    "mal_de_gorge":      (["mal de gorge", "rhinite", "nez qui coule"], False),
}

PNEUMO_KEYS = [k for k, (_, c) in SYMPTOMS.items() if c]
OTHER_KEYS = [k for k, (_, c) in SYMPTOMS.items() if not c]

NOTE_TEMPLATES = [
    "Patient presentant {s}.",
    "Consultation pour {s}.",
    "Motif : {s}.",
    "Depuis quelques jours : {s}.",
    "Note clinique : {s}.",
    "Le patient se plaint de {s}.",
    "Antecedents sans particularite. {s}.",
    "Enfant amene aux urgences pour {s}.",
]
NO_SYMPTOM_NOTES = [
    "Aucun symptome particulier.",
    "Patient asymptomatique, controle de routine.",
    "Pas de plainte respiratoire.",
    "RAS cliniquement.",
    "Bilan systematique, sans symptome.",
]


def join_fr(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


def fmt_pct(x):
    return f"{x:.1f}%"


def confidence_label(c):
    if c >= 85:
        return "haute"
    if c >= 65:
        return "moyenne"
    return "faible"


# ----------------------------------------------------------------------------
# Case simulation
# ----------------------------------------------------------------------------
def simulate_case(rng):
    true_label = rng.choices(["PNEUMONIA", "NORMAL"], weights=[0.62, 0.38])[0]  # Kaggle imbalance

    # CNN predicted class (mostly right, sometimes wrong)
    if rng.random() < 0.93:
        pred = true_label
    else:
        pred = "NORMAL" if true_label == "PNEUMONIA" else "PNEUMONIA"

    # confidence in the predicted class
    r = rng.random()
    if r < 0.60:
        conf = rng.uniform(90, 100)
    elif r < 0.85:
        conf = rng.uniform(70, 90)
    else:
        conf = rng.uniform(50, 70)
    conf = round(conf, 1)

    p_pneu = conf if pred == "PNEUMONIA" else 100 - conf
    p_pneu = round(p_pneu, 1)
    raw = round(min(100.0, max(0.0, p_pneu + rng.uniform(-2, 2))), 1)
    calibrated = p_pneu
    inside = round(rng.uniform(0.35, 0.85), 3)

    # clinical note (60% of cases have one)
    has_note = rng.random() < 0.60
    symptoms = []
    note = ""
    if has_note:
        mode = rng.choices(["pneumo", "mixed", "other", "none"], weights=[0.45, 0.2, 0.2, 0.15])[0]
        # make symptoms loosely correlated with the true label
        if true_label == "NORMAL" and mode == "pneumo" and rng.random() < 0.6:
            mode = rng.choice(["other", "none"])
        if true_label == "PNEUMONIA" and mode == "other" and rng.random() < 0.5:
            mode = "pneumo"
        if mode == "pneumo":
            symptoms = rng.sample(PNEUMO_KEYS, rng.randint(1, 4))
        elif mode == "mixed":
            symptoms = rng.sample(PNEUMO_KEYS, rng.randint(1, 2)) + rng.sample(OTHER_KEYS, 1)
        elif mode == "other":
            symptoms = rng.sample(OTHER_KEYS, rng.randint(1, 2))
        if symptoms:
            phrases = [rng.choice(SYMPTOMS[k][0]) for k in symptoms]
            note = rng.choice(NOTE_TEMPLATES).format(s=join_fr(phrases))
        else:
            note = rng.choice(NO_SYMPTOM_NOTES)

    return dict(true_label=true_label, pred=pred, conf=conf, p_pneu=p_pneu, raw=raw,
                calibrated=calibrated, inside=inside, symptoms=symptoms, note=note)


# ----------------------------------------------------------------------------
# Fusion / triage rules (the "ground truth" the text model should learn)
# ----------------------------------------------------------------------------
def fuse(case):
    pred, conf = case["pred"], case["conf"]
    syms = case["symptoms"]
    n_pneu = sum(1 for k in syms if SYMPTOMS[k][1])
    has_note = bool(case["note"])
    has_syms = len(syms) > 0

    if conf < 60:
        return "image_seule" if not has_note else "indetermine", "indetermine", "oui"

    if not has_note or not has_syms:
        coherence = "image_seule"
        triage = "pneumonie_suspecte" if pred == "PNEUMONIA" else "pneumonie_peu_probable"
        return coherence, triage, "non"

    if pred == "PNEUMONIA":
        if n_pneu >= 1:
            return "coherent", "pneumonie_suspecte_prioritaire" if n_pneu >= 3 else "pneumonie_suspecte", "non"
        return "discordant", "avis_medical_recommande", "non"
    else:
        if n_pneu >= 2:
            return "discordant", "avis_medical_recommande", "non"
        return "coherent", "pneumonie_peu_probable", "non"


# ----------------------------------------------------------------------------
# Report text
# ----------------------------------------------------------------------------
CLASS_FR = {"PNEUMONIA": "pneumonie", "NORMAL": "normal"}

OPENINGS = [
    "Classe favorisee par le modele : {label}.",
    "Le classifieur oriente vers : {label}.",
    "Prediction de l'image : {label}.",
    "Resultat de la classification : {label}.",
]
OPENINGS_REFUSED = [
    "Classe favorisee par le modele : {label}, mais la confiance est faible ({conf}). Resultat non fiable : le modele s'abstient.",
    "Le classifieur penche vers {label} avec une confiance faible ({conf}) : le modele prefere s'abstenir.",
    "Prediction {label} trop peu assuree ({conf}) pour conclure : le modele s'abstient.",
]
PROBA_TPL = [
    "Probabilite calibree estimee pour la classe pneumonie : {p} (ce n'est pas une probabilite clinique).",
    "Score calibre pour la pneumonie : {p} ; il ne s'agit pas d'une probabilite clinique.",
    "Pneumonie estimee a {p} apres calibration (indicateur du modele, pas de risque clinique).",
]
CONF_TPL = [
    "Confiance du modele dans cette classe : {c} ({lvl}).",
    "Niveau de confiance du modele : {c} ({lvl}).",
    "Le modele affiche une confiance de {c} ({lvl}) sur cette classe.",
]
GRADCAM_TPL = [
    "Grad-CAM : {p}% de l'activation est dans le masque thoracique, {zone}. {note}",
    "Carte Grad-CAM : {p}% de l'activation tombe dans le masque thoracique, {zone}. {note}",
    "Grad-CAM : activation a {p}% dans la zone thoracique, {zone}. {note}",
]
GRADCAM_NOTES = [
    "Cette carte montre des zones influentes pour le modele; elle ne localise ni ne confirme une lesion.",
    "La carte indique les regions qui ont pese sur la decision; elle ne prouve ni ne situe une lesion.",
    "Il s'agit d'une carte d'attention du modele, non d'une localisation ou d'une confirmation de lesion.",
]
ZONE_OK = ["coherente avec la zone thoracique", "dans le champ thoracique attendu", "en accord avec le masque thoracique"]
ZONE_KO = ["peu coherente avec la zone thoracique", "plutot hors du masque thoracique", "eloignee de la zone thoracique attendue"]
SYM_TPL = ["Symptomes cliniques releves : {s}.", "Symptomes extraits de la note : {s}.", "Note clinique : {s}."]
DISCORD_TPL = [
    "Discordance entre l'image et le tableau clinique : un avis medical est recommande.",
    "L'image et la clinique divergent : une relecture medicale est conseillee.",
    "Incoherence image/clinique : avis medical recommande.",
]
COHER_TPL = [
    "Le tableau clinique est coherent avec le resultat de l'image.",
    "Image et clinique concordent.",
    "Le resultat de l'image s'accorde avec le tableau clinique.",
]
FUSION_TPL = [
    "Fusion : {coh} ; resultat de triage experimental : {tri}.",
    "Decision de fusion : {coh} ; triage experimental : {tri}.",
    "Resultat de fusion : {coh} ; niveau de triage : {tri}.",
]
NONOTE_TPL = [
    "Symptomes cliniques non renseignes : l'image ne permet pas de les determiner. Ajoutez une note clinique pour analyser les symptomes, s'il y en a.",
    "Aucune note clinique fournie : le modele ne peut pas etablir de symptomes a partir de l'image. Ajoutez une note pour les analyser.",
]
NOSYM_TPL = ["Aucun symptome pertinent releve dans la note clinique.", "La note ne mentionne pas de symptome pertinent."]
DISCLAIMERS = [
    "Prototype pedagogique, ne constitue pas un diagnostic medical.",
    "Rappel : ce prototype pedagogique ne constitue pas un diagnostic medical.",
    "Avertissement : outil pedagogique, ne constitue pas un diagnostic medical.",
]


def build_report(case, coherence, triage, refused, rng):
    pred_fr = CLASS_FR[case["pred"]]
    pct_in = round(case["inside"] * 100)
    zone = rng.choice(ZONE_OK if pct_in >= 50 else ZONE_KO)
    parts = []

    if refused == "oui":
        parts.append(rng.choice(OPENINGS_REFUSED).format(
            label=pred_fr, conf=fmt_pct(case["conf"])))
    else:
        parts.append(rng.choice(OPENINGS).format(label=pred_fr))
        parts.append(rng.choice(PROBA_TPL).format(p=fmt_pct(case["calibrated"])))
        parts.append(rng.choice(CONF_TPL).format(
            c=fmt_pct(case["conf"]), lvl=confidence_label(case["conf"])))

    parts.append(rng.choice(GRADCAM_TPL).format(
        p=pct_in, zone=zone, note=rng.choice(GRADCAM_NOTES)))

    if case["symptoms"]:
        names = join_fr([k.replace("_", " ") for k in case["symptoms"]])
        parts.append(rng.choice(SYM_TPL).format(s=names))
        if coherence == "discordant":
            parts.append(rng.choice(DISCORD_TPL))
        elif coherence == "coherent":
            parts.append(rng.choice(COHER_TPL))
    elif case["note"]:
        parts.append(rng.choice(NOSYM_TPL))
    else:
        parts.append(rng.choice(NONOTE_TPL))

    parts.append(rng.choice(FUSION_TPL).format(coh=coherence, tri=triage))
    parts.append(rng.choice(DISCLAIMERS))
    return " ".join(parts)


def build_input_text(case):
    """Single text prompt: what a text model would receive."""
    return (
        f"classe_cnn={CLASS_FR[case['pred']]} | proba_pneumonie_calibree={case['calibrated']} | "
        f"confiance={case['conf']} | gradcam_inside_ratio={case['inside']} | "
        f"note_clinique={case['note'] or 'AUCUNE'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    for i in range(args.n):
        c = simulate_case(rng)
        coherence, triage, refused = fuse(c)
        report = build_report(c, coherence, triage, refused, rng)
        rows.append({
            "id": i,
            "kaggle_label": c["true_label"],
            "cnn_pred": c["pred"],
            "p_pneumonia_raw": c["raw"],
            "p_pneumonia_calibrated": c["calibrated"],
            "confidence": c["conf"],
            "confidence_level": confidence_label(c["conf"]),
            "gradcam_inside_ratio": c["inside"],
            "clinical_note": c["note"],
            "symptoms": ";".join(c["symptoms"]),
            "coherence": coherence,
            "triage": triage,
            "refused": refused,
            "input_text": build_input_text(c),
            "report": report,
        })

    with open("reports_dataset.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("reports_dataset.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows written.")


if __name__ == "__main__":
    main()