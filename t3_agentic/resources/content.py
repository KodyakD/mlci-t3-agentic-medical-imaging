GUIDELINES_PNEUMONIA = (
    "Radiographie thoracique, reperes de lecture (prototype pedagogique):\n"
    "- une opacite du lobe inferieur chez un patient symptomatique evoque une pneumonie;\n"
    "- l'absence de symptomes respiratoires n'exclut pas une lesion;\n"
    "- la qualite de l'image compte plus que la puissance du modele;\n"
    "- le modele ne distingue pas bacterien/viral et ne localise pas une lesion;\n"
    "- il ne peut pas remplacer la lecture d'un radiologue.\n"
    "Si l'image et la note clinique se contredisent, dire l'incoherence au lieu de trancher."
)

PROTOTYPE_WARNING = (
    "Prototype pedagogique. Ne constitue pas un diagnostic medical.\n"
    "Systeme experimental de TP : aucune decision clinique ne doit reposer sur lui."
)

EXAMPLES = [
    {"id": "case_01", "ask": "Que peut-on dire de cette radio ?", "route": "image"},
    {"id": "case_02", "ask": "Analyse cette note : toux productive et fievre elevee depuis 5 jours.",
     "route": "text"},
    {"id": "case_03", "ask": "Patient avec cette radio et cette note : toux + fievre.",
     "route": "both"},
]
