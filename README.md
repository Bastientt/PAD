# 🛡️ PAD - Presence & Liveness Detection

**PAD** est un système de sécurité biométrique conçu pour vérifier la présence réelle d'un utilisateur (Liveness Detection). Il utilise une analyse de pose de tête en 3D pour valider des défis de mouvements aléatoires (ex: GAUCHE, HAUT, BAS), empêchant ainsi les fraudes par photo ou vidéo pré-enregistrée.

---

## 🏗️ Architecture Technique

Le projet est découpé en micro-services orchestrés par Docker pour garantir une isolation granulaire :

* **Frontend** : Interface utilisateur (React Native/React) pour la capture vidéo et l'affichage des défis.
* **Backend (Rust)** : API haute performance (Actix-web) gérant les sessions, les défis et la communication asynchrone.
* **IA_Worker (Python)** : Moteur d'analyse basé sur **MediaPipe** et **OpenCV**. Il estime la rotation de la tête via l'algorithme `solvePnP`.
* **Infrastructure** :
    * **Redis** : Bus de messages (Pub/Sub) pour la distribution des jobs d'analyse.
    * **MinIO** : Stockage S3-compatible pour les vidéos temporaires avant analyse.

---

## 🧠 Comment marche l'IA ?

L'analyse transforme un flux vidéo en données télémétriques précises pour une prise de décision binaire (Validation du défi).



### 1. Acquisition et Masquage
* **Traitement en RAM** : La vidéo est téléchargée depuis MinIO directement en mémoire vive pour optimiser la vitesse de traitement.
* **Extraction des Landmarks** : Le worker applique un masque via **MediaPipe Face Mesh** pour extraire 468 points de repère faciaux en 3D.
* **Isolation Bio-métrique** : Nous isolons des points critiques (nez, menton, yeux et bouche) pour créer une signature géométrique du visage.

### 2. Télémétrie et Vectorisation (SolvePnP)
L'algorithme `solvePnP` compare les points 2D extraits de l'image avec un modèle 3D théorique.
* **Vectorisation** : Les déplacements de ces points sont convertis en une liste télémétrique temporelle (angles de rotation).
* **Angles d'Euler** : On extrait le **Yaw** (rotation) et le **Pitch** (inclinaison) :
    * $$Yaw = \arctan2(R_{0,2}, R_{2,2})$$
    * $$Pitch = \arcsin(-R_{1,2})$$

### 3. Traitement du Signal et Lissage
* **Lissage** : Les données subissent un lissage (Filtre One Euro) pour éliminer le bruit et les tremblements du capteur.
* **Seuil de Tolérance (Ratio)** : Un paramètre de configuration (`RATIO`) définit la sensibilité de détection (par défaut **0.60**).

### 4. Validation par Hystérésis (Machine à états)
Le système utilise une machine à états à double seuil pour valider le challenge :
* **Zone Neutre** : L'IA impose un retour au centre entre chaque mouvement pour éviter les validations accidentelles.
* **Validation Séquentielle** : Le défi n'est réussi que si la séquence détectée correspond exactement au challenge généré.



---

## 📡 API Endpoints (Backend Rust)

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/challenge/new` | Génère une séquence aléatoire et un `user_id`. |
| `POST` | `/video/upload` | Upload vers MinIO et publie le job dans la queue Redis `ia_jobs`. |
| `GET` | `/result/{user_id}` | Récupère le verdict final de l'analyse. |

### Format de sortie Redis (`ia_results`) :
La télémétrie est formatée en une chaîne brute séparée par des virgules pour un traitement simplifié par le middleware Rust.
```json
{
  "user_id": "4643453044896821454",
  "status": "IA_SUCCESS",
  "filename": "video_8712.mp4",
  "movements": "BAS,GAUCHE,BAS,HAUT"
}