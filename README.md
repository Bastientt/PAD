# 🛡️ PAD - Presence & Liveness Detection

**PAD** est un système de sécurité biométrique conçu pour vérifier la présence réelle d'un utilisateur (Liveness Detection). Il utilise une analyse de pose de tête en 3D pour valider des défis de mouvements aléatoires (ex: GAUCHE, HAUT, BAS), empêchant ainsi les fraudes par photo ou vidéo pré-enregistrée.



---

## 🏗️ Architecture Technique

Le projet est découpé en micro-services orchestrés par Docker :

* **Frontend** : Interface utilisateur pour la capture vidéo.
* **Backend (Rust)** : API haute performance (Actix-web) gérant les sessions, les défis et la communication asynchrone via Redis.
* **IA_Worker (Python)** : Moteur d'analyse basé sur **MediaPipe** et **OpenCV**. Il estime la rotation de la tête en degrés réels via l'algorithme `solvePnP`.
* **Infrastructure** :
    * **Redis** : Bus de messages (Pub/Sub) pour la distribution des jobs d'analyse.
    * **MinIO** : Stockage S3-compatible pour les vidéos temporaires avant analyse.

---

## 🧠 Comment marche l'IA ?

L'analyse ne repose pas sur une simple reconnaissance d'image, mais sur une reconstruction géométrique du visage :

### 1. Extraction des Landmarks
Le worker utilise MediaPipe Face Mesh pour extraire 468 points de repère faciaux en 3D. Pour le calcul de pose, nous isolons 6 points critiques : le bout du nez, le menton, les coins externes des yeux et les coins de la bouche.

### 2. Estimation de Pose 3D (SolvePnP)
L'algorithme `solvePnP` (Perspective-n-Point) compare ces points 2D extraits de l'image avec un modèle de visage 3D générique. Cette méthode permet de calculer une matrice de rotation et de s'affranchir des distorsions liées à la distance entre l'utilisateur et son téléphone.

### 3. Conversion en Angles d'Euler
La matrice de rotation est convertie en degrés réels pour obtenir le **Yaw** (rotation gauche/droite) et le **Pitch** (inclinaison haut/bas) :
* **Yaw** : $\arctan2(R_{0,2}, R_{2,2})$
* **Pitch** : $\arcsin(-R_{1,2})$



### 4. Validation par Hystérésis
Pour garantir une détection robuste, le worker utilise une machine à états à double seuil :
* **Détection** : Le mouvement est validé si l'angle dépasse un seuil (ex: 20°) pendant plusieurs frames consécutives.
* **Retour au Neutre** : L'étape suivante du défi ne se débloque que si l'utilisateur revient dans une zone centrale (zone de sécurité), empêchant ainsi les validations multiples d'un même mouvement.

---

## 📡 API Endpoints (Backend Rust)

L'API communique avec le frontend et délègue l'analyse lourde au worker via Redis.

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/challenge/new` | Génère une séquence aléatoire (ex: `GAUCHE,HAUT`) et un `user_id` temporaire. |
| `POST` | `/video/upload` | Reçoit le fichier `.mp4`/`.webm`. Upload la vidéo sur MinIO et publie un job dans la queue Redis `ia_jobs`. |
| `GET` | `/result/{user_id}` | Vérifie le statut de l'analyse (Polling ou via WebSocket). |

### Exemple de payload de retour (Redis `ia_results`) :
```json
{
  "user_id": "8708301549686279141",
  "status": "IA_SUCCESS",
  "filename": "video_123.mp4",
  "details": "Séquence détectée: ['GAUCHE', 'HAUT']"
}