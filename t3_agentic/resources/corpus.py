"""Small reference corpus for the RAG tool (in French: the agent's working language).

The passages are written for this TP (no third-party copyrighted text) and are
deliberately short: the goal is to show *retrieval-then-answer* over an MCP
resource, not to build a clinical knowledge base.
"""
CORPUS = [
    {"id": "g01", "title": "Lecture systematique d'une radiographie thoracique",
     "tags": ["methode", "lecture", "radiographie"],
     "text": "Une lecture systematique verifie d'abord la qualite technique (inspiration, rotation, "
             "penetrabilite), puis les voies aériennes, les champs pulmonaires, la plèvre, le "
             "mediastin et les os. Une consolidation lobaire floute souvent le diaphragme et le "
             "bord cardiaque, les lobes superieurs etant relativement epargnes sur le cliché de face."},
    {"id": "g02", "title": "Signes radiologiques de la pneumonie",
     "tags": ["signes", "pneumonie", "consolidation", "epanchement"],
     "text": "La pneumonie se traduit le plus souvent par une opacite alvéolaire (consolidation) "
             "lobaire ou segmentaire avec bronchogrammes aeriques, parfois associee a un epanchement "
             "pleural. Sur un cliché de face, la silhouettisation du bord cardiaque evoque une "
             "atteinte du lobe moyen et celle du diaphragme une atteinte des lobes inferieurs."},
    {"id": "g03", "title": "Radiographies pediatriques et adultes",
     "tags": ["pediatrie", "pediatrique", "adulte", "adulte", "radio", "radiographie", "domaine", "differencier"],
     "text": "Les radiographies thoraciques pediatriques different fortement des adultes : thorax "
             "plus petit, diaphragme plus plat, silhouette cardiaque proportionnellement plus "
             "grande, ombre thymique frequente. Un modele appris sur des images pediatriques se "
             "transfere donc mal aux adultes et son comportement doit etre calibre par population."},
    {"id": "g04", "title": "Projection et acquisition",
     "tags": ["acquisition", "projection", "appareil"],
     "text": "La majorite des radiographies portables sont en incidence AP, patient debout ou "
             "couche, alors que les incidences PA sont realisees debout. Differences de projection, "
             "d'appareil et de post-traitement constituent une source de decalage de domaine qu'aucune "
             "normalisation image ne supprime."},
    {"id": "g05", "title": "Ce qu'un CNN peut et ne peut pas conclure",
     "tags": ["cnn", "limites", "modele"],
     "text": "Un classifieur binaire entrain sur des radiographies thoraciques sort une probabilite "
             "d'opacite de type pneumonie. Il ne localise pas la lesion, ne distingue pas bacterien "
             "et viral, n'evalue pas la gravite et ne remplace pas un compte rendu radiologique."},
    {"id": "g06", "title": "Lire une carte de chaleur Grad-CAM",
     "tags": ["gradcam", "explicabilite", "heatmap"],
     "text": "Une carte Grad-CAM met en evidence les regions qui ont le plus influence la "
             "prediction. C'est un indicateur de plausibilite, pas une explication causale : si la zone "
             "surlignee est exterieure au thorax (bord, marqueur, texte), l'explication doit etre "
             "consideree comme non fiable et la confiance abaissee."},
    {"id": "g07", "title": "Pourquoi les labels NIH sont bruyants",
     "tags": ["labels", "nih", "qualite"],
     "text": "Les labels publics de la Chest X-ray14 du NIH ont ete extraits par traitement "
             "automatique du language des comptes rendus : ils sont multi-labeles et imparfaits. Un "
             "compte rendu mentionnant 'pas de condensation focale' peut encore produire un label "
             "positif. Les cas etiquetes 'Infiltration' ne sont pas des pneumonies au sens clinique."},
    {"id": "g08", "title": "Decalage de domaine et ses consequences",
     "tags": ["domaine", "generalisation", "calibration"],
     "text": "Un modele evalue hors de sa population d'entrainement peut conserver un bon classement "
             "(AUC) alors que son exactitude s'effondre, car le seuil de decision a ete ajuste "
             "ailleurs. La maniere honnete de rapporter le resultat est l'accuracy equilibree, le rappel "
             "par classe et la couverture apres le seuil de confiance, en explicitant ce seuil."},
    {"id": "g09", "title": "Contexte clinique d'une radio",
     "tags": ["clinique", "contexte", "symptomes"],
     "text": "Fievre, toux productive, frissons, dyspnee, douleur thoracique et syndrome "
             "inflammatoire eleve orientent l'interpretation d'une opacite. L'absence de symptomes "
             "diminue la probabilite de pneumonie sans l'eliminer, et une image seule n'etablit pas "
             "le diagnostic."},
    {"id": "g10", "title": "Signes d'alerte et escalade",
     "tags": ["securite", "triage", "urgence"],
     "text": "Des signes evocant une urgence (epanchement important, pneumothorax, cavitation, "
             "hypoxemie marquee, signes de sepsis) doivent etre remontes immediatement a un clinicien. "
             "Un prototype automatise ne doit jamais retarder cette escalade, il peut seulement "
             "l'appuyer."},
    {"id": "g11", "title": "Ethique d'un prototype d'imagerie",
     "tags": ["ethique", "limites", "biais"],
     "text": "Un prototype entraine sur des donnees publiques herite de leurs biais : il sera plus "
             "fiable sur des populations proches des donnees d'entrainement et moins fiable ailleurs. "
             "Chaque sortie doit porter un avertissement non diagnostique et le systeme doit savoir "
             "s'abstenir plutot que d'imposer une etiquette."},
    {"id": "g12", "title": "Rapporter ce qu'un systeme a fait",
     "tags": ["rapport", "methode", "evaluation"],
     "text": "Un rapport doit indiquer la provenance des donnees, le decoupage (niveau patient), "
             "les metriques avec leurs intervalles de confiance, le point de fonctionnement et sa "
             "methode d'ajustement, l'ablation de chaque composant, et au moins cinq echecs analyses. "
             "Des chiffres sans leur protocole ne sont pas des resultats."},
]
