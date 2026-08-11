import os
import numpy as np
import keras
from keras.utils import load_img, img_to_array
from flask import Flask, request, jsonify
from flask_cors import CORS

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
MODELE_PATH = "meilleur_modele_plantes.h5"
CLASSES_PATH = "classes.txt"
PORT = 5000
HOST = "0.0.0.0"  # écoute sur toutes les interfaces réseau (nécessaire pour Flutter)

# =====================================================================
# 2. CHARGEMENT DU MODELE (une seule fois, au démarrage du serveur)
# =====================================================================
if not os.path.exists(MODELE_PATH):
    raise FileNotFoundError(f"Le fichier du modèle '{MODELE_PATH}' n'existe pas.")

print("Chargement du modèle entraîné... (Veuillez patienter)")

try:
    model = keras.models.load_model(MODELE_PATH, compile=False)
except Exception as e:
    print(f"Premier essai échoué ({e}), nouvel essai avec les classes internes Keras 3...")
    from keras.src.ops.numpy import TrueDivide, Subtract
    model = keras.models.load_model(
        MODELE_PATH,
        compile=False,
        custom_objects={"TrueDivide": TrueDivide, "Subtract": Subtract},
    )

print("✅ Modèle chargé avec succès.")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Le fichier des classes '{CLASSES_PATH}' n'existe pas.")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"Nombre de classes chargées : {len(class_names)}")
print(f"Nombre de sorties du modèle : {model.output_shape[-1]}")
if len(class_names) != model.output_shape[-1]:
    print("⚠️  ATTENTION : classes.txt ne correspond pas au nombre de sorties du modèle.")

# =====================================================================
# 3. DICTIONNAIRE DES TRAITEMENTS
# =====================================================================
conseils_maladies = {
  "Pepper__bell___Bacterial_spot": {"traitement": "Utiliser un fongicide bactéricide et enlever les feuilles infectées.", "prevention": "Assurer une bonne circulation d’air et éviter les arrosages nocturnes."},
  "Pepper__bell___healthy": {"traitement": "Aucun traitement nécessaire.", "prevention": "Maintenir une bonne hygiène et pratiquer la rotation des cultures."},
  "PlantVillage": {"traitement": "Non applicable.", "prevention": "N/A"},
  "Potato___Early_blight": {"traitement": "Fongicide à base de chlorothalonil ou cuivre au début des symptômes.", "prevention": "Éviter l’humidité prolongée sur le feuillage. Rotation des cultures."},
  "Potato___Late_blight": {"traitement": "Fongicides systémiques dès les premières taches.", "prevention": "Détruire les résidus, variétés résistantes."},
  "Potato___healthy": {"traitement": "Aucun traitement nécessaire.", "prevention": "Bonnes pratiques agricoles."},
  "Tomato_Bacterial_spot": {"traitement": "Appliquer des produits à base de cuivre.", "prevention": "Nettoyage des outils, éviter les éclaboussures."},
  "Tomato_Early_blight": {"traitement": "Fongicide dès l’apparition des premiers signes.", "prevention": "Paillage, rotation."},
  "Tomato_Late_blight": {"traitement": "Fongicides systémiques urgents.", "prevention": "Éliminer les plantes atteintes, contrôler l’humidité."},
  "Tomato_Leaf_Mold": {"traitement": "Fongicides adaptés et diminution de l’humidité.", "prevention": "Aération, éviter la brumisation."},
  "Tomato_Septoria_leaf_spot": {"traitement": "Fongicide contenant mancozèbe ou cuivre.", "prevention": "Enlever feuilles mortes, espacer les plants."},
  "Tomato_Spider_mites_Two_spotted_spider_mite": {"traitement": "Insecticides ciblant les acariens ou savon insecticide.", "prevention": "Pulvérisation d’eau pour réduire la chaleur."},
  "Tomato__Target_Spot": {"traitement": "Fongicide à base de chlorothalonil.", "prevention": "Éviter l’humidité sur le feuillage."},
  "Tomato__Tomato_YellowLeaf__Curl_Virus": {"traitement": "Incurable — enlever les plants infectés.", "prevention": "Contrôler les mouches blanches, variétés résistantes."},
  "Tomato__Tomato_mosaic_virus": {"traitement": "Détection et destruction des plantes infectées.", "prevention": "Désinfection du matériel, variétés résistantes."},
  "Tomato_healthy": {"traitement": "Aucun traitement nécessaire.", "prevention": "Maintenir un environnement sain."},
  "Soybean_Bacterial_blight": {"traitement": "Cuivre ou antibiotiques autorisés au stade initial.", "prevention": "Utiliser semences saines et variétés résistantes, rotation des cultures."},
  "Soybean_Mosaic_Virus": {"traitement": "Retirer les plants infectés.", "prevention": "Semences certifiées, contrôler les vecteurs (pucerons)."},
  "Corn_Gray_Leaf_Spot": {"traitement": "Fongicides foliaires en phase initiale.", "prevention": "Rotation des cultures, gestion des résidus."},
  "Corn_Southern_Leaf_Blight": {"traitement": "Fungicides pendant la floraison.", "prevention": "Variétés résistantes, enfouissement des résidus."},
  "Millet_Blast": {"traitement": "Application de fongicides au stade initial.", "prevention": "Variétés résistantes, espacement optimal, drainage."},
  "Millet_Rust_Downy_Smut_Ergot": {"traitement": "Produits biologiques ou chimiques selon contexte.", "prevention": "Rotation, semences saines, destruction des résidus."}
}

# =====================================================================
# 4. FONCTION DE PRÉDICTION (réutilisable)
# =====================================================================
def predire_maladie(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]

    indice_max = int(np.argmax(score))
    classe_predite = class_names[indice_max]
    confiance = float(100 * np.max(score))

    infos = conseils_maladies.get(
        classe_predite,
        {"traitement": "Aucun traitement spécifique enregistré.",
         "prevention": "Pas de mesures spécifiques enregistrées."}
    )

    return {
        "classe": classe_predite,
        "confiance": round(confiance, 2),
        "traitement": infos["traitement"],
        "prevention": infos["prevention"],
    }

# =====================================================================
# 5. SERVEUR FLASK
# =====================================================================
app = Flask(__name__)
CORS(app)  # autorise les requêtes venant de l'app Flutter

UPLOAD_DIR = "uploads_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    """Permet à l'app Flutter de vérifier que le serveur est joignable."""
    return jsonify({"status": "ok", "classes": len(class_names)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principal.
    Flutter doit envoyer une requête multipart/form-data
    avec un champ nommé "image" contenant le fichier photo.
    """
    if "image" not in request.files:
        return jsonify({"error": "Aucun fichier reçu. Le champ doit s'appeler 'image'."}), 400

    fichier = request.files["image"]
    if fichier.filename == "":
        return jsonify({"error": "Nom de fichier vide."}), 400

    chemin_temp = os.path.join(UPLOAD_DIR, fichier.filename)
    fichier.save(chemin_temp)

    try:
        resultat = predire_maladie(chemin_temp)
        return jsonify(resultat), 200
    except Exception as e:
        return jsonify({"error": f"Erreur pendant la prédiction : {str(e)}"}), 500
    finally:
        if os.path.exists(chemin_temp):
            os.remove(chemin_temp)


# =====================================================================
# 6. MAIN
# =====================================================================
if __name__ == "__main__":
    print(f"\n🚀 Serveur lancé sur http://{HOST}:{PORT}")
    print("   - Depuis un émulateur Android : utilisez http://10.0.2.2:5000")
    print("   - Depuis un téléphone physique : utilisez l'adresse IP locale de ce PC (ex: http://192.168.1.X:5000)")
    print("   - Testez avec : curl http://127.0.0.1:5000/health\n")
    app.run(host=HOST, port=PORT, debug=False)