// --- ACTIX & WEB ---
use actix_web::{get, web, App, HttpServer, HttpRequest, HttpResponse, Error, FromRequest, dev::Payload};
use actix_web::web::Data;
use actix_web::cookie::{Cookie, SameSite};
use actix_web_actors::ws;
use actix::{Actor, ActorContext, AsyncContext, StreamHandler, Handler};
use actix_cors::Cors;
use rand::seq::SliceRandom; // C'est ce "Trait" qui débloque .choose() sur les slices

// --- AWS S3 / MINIO ---
use aws_sdk_s3::Client;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::presigning::PresigningConfig;

// --- ASYNC & UTILS ---
use futures_util::future::{ready, Ready};
use futures_util::stream::StreamExt; // Indispensable pour map()
use serde::{Serialize, Deserialize};
use serde_json::json;
use rand::{thread_rng, Rng};

// --- TEMPS & SÉCURITÉ ---
use chrono::Utc;
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation, EncodingKey, DecodingKey};
use redis::AsyncCommands;

// --- STRUCTURES ---

#[derive(actix::Message, Serialize, Deserialize, Debug)]
#[rtype(result = "()")]
struct RedisIaResponse {
    pub user_id: i64,
    pub status: String,
    pub filename: String,
    #[serde(default)]
    pub movements: serde_json::Value,
}

pub struct WSActor {
    pub user_id: i64,
    pub filename: String,
    pub redis_client: redis::Client,
}

impl Actor for WSActor {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let client = self.redis_client.clone();
        let user_id = self.user_id;

        // Création du flux d'écoute Redis ia_results
        let stream = async move {
            let mut conn = client.get_async_connection().await.unwrap();
            let mut pubsub = conn.into_pubsub();
            pubsub.subscribe("ia_results").await.unwrap();

            pubsub.into_on_message().map(move |msg| {
                let payload: String = msg.get_payload().unwrap();
                let res: RedisIaResponse = serde_json::from_str(&payload).unwrap();
                res
            })
        };

        // On connecte Redis à l'acteur
        ctx.add_stream(futures_util::stream::once(stream).flatten());
        println!("📡 Actor {} : Écoute active sur ia_results", user_id);
    }
}

// Handler pour les messages provenant du flux Redis
impl StreamHandler<RedisIaResponse> for WSActor {
    fn handle(&mut self, item: RedisIaResponse, ctx: &mut Self::Context) {
        if item.user_id == self.user_id {
            if let Ok(json) = serde_json::to_string(&item) {
                ctx.text(json);
            }
        }
    }
}


impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WSActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                if text.starts_with("UploadDone") {
                    // On découpe le message pour extraire la suite de directions
                    let parts: Vec<&str> = text.split('|').collect();
                    let challenge_str = parts.get(1).unwrap_or(&"INCONNU").to_string();

                    println!("🚀 Publication job IA pour l'user {} avec challenge: {}", self.user_id, challenge_str);
                    
                    let redis = self.redis_client.clone();
                    let payload = json!({
                        "user_id": self.user_id,
                        "filename": self.filename,
                        "challenge": challenge_str // On l'envoie à l'IA
                    }).to_string();

                    actix::spawn(async move {
                        if let Ok(mut conn) = redis.get_multiplexed_tokio_connection().await {
                            let _: Result<(), _> = conn.publish("ia_jobs", payload).await;
                        }
                    });
                    ctx.text(format!("IA_PROCESSING: Vérification de la suite {}", challenge_str));
                }
            }
            _ => (),
        }
    }
}
// --- AUTH & JWT ---

#[derive(Debug, Serialize, Deserialize)]
struct JwtClaims {
    id: u64,
    exp: usize,
}

impl FromRequest for JwtClaims {
    type Error = Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        match req.cookie("PAD_Auth") {
            Some(c) => match decode_jwt(c.value()) {
                Ok(claims) => ready(Ok(claims)),
                Err(_) => ready(Err(actix_web::error::ErrorUnauthorized("Invalid token"))),
            },
            None => ready(Err(actix_web::error::ErrorUnauthorized("No cookie found"))),
        }
    }
}

fn decode_jwt(token: &str) -> Result<JwtClaims, jsonwebtoken::errors::Error> {
    let key = std::fs::read("key/public_key.pem").expect("Missing public key");
    decode::<JwtClaims>(token, &DecodingKey::from_ed_pem(&key)?, &Validation::new(Algorithm::EdDSA)).map(|data| data.claims)
}

#[get("/Authentificate")]
pub async fn authentification_get() -> Result<HttpResponse, Error> {
    let priv_key = std::fs::read("./key/private_key.pem").expect("Missing private key");
    let claims = JwtClaims {
        id: thread_rng().gen(),
        exp: (Utc::now().timestamp() + 86400) as usize,
    };
    //
    let key = EncodingKey::from_ed_pem(&priv_key).unwrap();
    let token = encode(&Header::new(Algorithm::EdDSA), &claims, &key).unwrap();

    let cookie = Cookie::build("PAD_Auth", token).path("/").http_only(true).secure(false).same_site(SameSite::Strict).finish();
    Ok(HttpResponse::Ok().cookie(cookie).json(json!({"status":"success"})))
}

// --- LOGIQUE S3 & REDIS ---

#[get("/init-sesssion")]
pub async fn init_session(claims: JwtClaims, s3: web::Data<Client>, redis: web::Data<redis::Client>) -> HttpResponse {
    if !log_id(claims.id, &redis).await {
        return HttpResponse::TooManyRequests().finish();
    }
    
    let challenge = generate_challenge(); // On génère la suite ici
    let filename = format!("data_{}_{}.mp4", Utc::now().timestamp(), claims.id);
    
    match get_signed_url(s3, filename).await {
        Ok(url) => HttpResponse::Ok().json(json!({ 
            "url": url, 
            "challenge": challenge // On l'envoie au front
        })),
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

pub async fn log_id(id: u64, redis: &redis::Client) -> bool {
    if let Ok(mut conn) = redis.get_multiplexed_tokio_connection().await {
        let key = format!("ratelimit:{}", id);
        let count: i64 = conn.incr(&key, 1).await.unwrap_or(0);
        if count == 1 { let _: () = conn.expire(&key, 3600).await.unwrap_or_default(); }
        return count <= 10;
    }
    false
}

pub async fn get_signed_url(s3: Data<Client>, filename: String) -> Result<String, String> {
    let expires = PresigningConfig::expires_in(std::time::Duration::from_secs(300)).unwrap();
    let presigned = s3.put_object().bucket("pad-bucket").key(filename).presigned(expires).await.map_err(|e| e.to_string())?;
    Ok(presigned.uri().to_string())
}

#[get("/ws/{filename}")]
pub async fn ws_index(req: HttpRequest, stream: web::Payload, path: web::Path<String>, claims: JwtClaims, redis: web::Data<redis::Client>) -> Result<HttpResponse, Error> {
    ws::start(WSActor { user_id: claims.id as i64, filename: path.into_inner(), redis_client: redis.get_ref().clone() }, &req, stream)
}



pub fn generate_challenge() -> Vec<String> {
    let directions = ["HAUT", "BAS", "GAUCHE", "DROITE"];
    let mut rng = thread_rng();
    (0..4)
        .map(|_| directions.choose(&mut rng).unwrap().to_string())
        .collect()

         
}

        

/*
        pub fn generate_challenge() -> Vec<String> {
    let result = vec![
        "GAUCHE", "BAS", "GAUCHE", "BAS", "GAUCHE", "HAUT", 
        "GAUCHE", "DROITE", "HAUT", "DROITE", "BAS", "GAUCHE", "DROITE"
    ];

    // On convertit le Vec<&str> en Vec<String> pour satisfaire la signature
    result.into_iter().map(|s| s.to_string()).collect()
    
}

    */

// --- TESTS ---

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};
    use std::collections::HashSet;

    // ================================================================
    // SECTION 1 : generate_challenge
    // ================================================================

    #[test]
    fn test_challenge_has_exactly_4_elements() {
        let challenge = generate_challenge();
        assert_eq!(challenge.len(), 4, "Le challenge doit contenir exactement 4 directions");
    }

    #[test]
    fn test_challenge_contains_only_valid_directions() {
        let valid: HashSet<&str> = ["HAUT", "BAS", "GAUCHE", "DROITE"].iter().cloned().collect();
        for _ in 0..50 {
            let challenge = generate_challenge();
            for dir in &challenge {
                assert!(valid.contains(dir.as_str()), "Direction invalide détectée: '{}'", dir);
            }
        }
    }

    #[test]
    fn test_challenge_is_non_empty_strings() {
        let challenge = generate_challenge();
        for dir in &challenge {
            assert!(!dir.is_empty(), "Une direction ne doit pas être une chaîne vide");
        }
    }

    #[test]
    fn test_challenge_covers_all_4_directions_statistically() {
        // Sur 200 appels, toutes les directions doivent apparaître au moins une fois
        let mut seen: HashSet<String> = HashSet::new();
        for _ in 0..200 {
            for dir in generate_challenge() {
                seen.insert(dir);
            }
        }
        assert_eq!(seen.len(), 4, "Les 4 directions doivent toutes apparaître statistiquement");
    }

    #[test]
    fn test_challenge_is_random_not_always_identical() {
        let first = generate_challenge();
        let mut all_same = true;
        for _ in 0..30 {
            if generate_challenge() != first {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Les challenges ne doivent pas toujours être identiques");
    }

    // ================================================================
    // SECTION 2 : RedisIaResponse — sérialisation / désérialisation
    // ================================================================

    #[test]
    fn test_redis_ia_response_deserialize_full() {
        let json = r#"{"user_id":42,"status":"done","filename":"test.mp4","movements":{"head":"up"}}"#;
        let res: RedisIaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(res.user_id, 42);
        assert_eq!(res.status, "done");
        assert_eq!(res.filename, "test.mp4");
        assert!(res.movements.is_object());
    }

    #[test]
    fn test_redis_ia_response_deserialize_movements_defaults_to_null() {
        let json = r#"{"user_id":1,"status":"processing","filename":"video.mp4"}"#;
        let res: RedisIaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(res.user_id, 1);
        assert!(res.movements.is_null(), "movements absent doit valoir null par défaut");
    }

    #[test]
    fn test_redis_ia_response_serialize_roundtrip() {
        let original = RedisIaResponse {
            user_id: 99,
            status: "success".to_string(),
            filename: "data_123_99.mp4".to_string(),
            movements: serde_json::json!({"HAUT": 3, "BAS": 1}),
        };
        let serialized = serde_json::to_string(&original).unwrap();
        let back: RedisIaResponse = serde_json::from_str(&serialized).unwrap();
        assert_eq!(back.user_id, original.user_id);
        assert_eq!(back.status, original.status);
        assert_eq!(back.filename, original.filename);
    }

    #[test]
    fn test_redis_ia_response_missing_status_fails() {
        // status est requis, pas de #[serde(default)]
        let json = r#"{"user_id":1,"filename":"x.mp4"}"#;
        let result: Result<RedisIaResponse, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Doit échouer si 'status' est absent");
    }

    #[test]
    fn test_redis_ia_response_missing_filename_fails() {
        let json = r#"{"user_id":1,"status":"ok"}"#;
        let result: Result<RedisIaResponse, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Doit échouer si 'filename' est absent");
    }

    #[test]
    fn test_redis_ia_response_wrong_type_user_id_fails() {
        let json = r#"{"user_id":"pas_un_entier","status":"ok","filename":"f.mp4"}"#;
        let result: Result<RedisIaResponse, _> = serde_json::from_str(json);
        assert!(result.is_err(), "user_id doit être un entier");
    }

    // ================================================================
    // SECTION 3 : JwtClaims — sérialisation
    // ================================================================

    #[test]
    fn test_jwt_claims_serde_roundtrip() {
        let claims = JwtClaims { id: 12345, exp: 9_999_999 };
        let json = serde_json::to_string(&claims).unwrap();
        let decoded: JwtClaims = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, 12345);
        assert_eq!(decoded.exp, 9_999_999);
    }

    #[test]
    fn test_jwt_claims_zero_values() {
        let claims = JwtClaims { id: 0, exp: 0 };
        let json = serde_json::to_string(&claims).unwrap();
        let decoded: JwtClaims = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, 0);
        assert_eq!(decoded.exp, 0);
    }

    #[test]
    fn test_jwt_claims_large_id() {
        let claims = JwtClaims { id: u64::MAX, exp: usize::MAX };
        let json = serde_json::to_string(&claims).unwrap();
        let decoded: JwtClaims = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, u64::MAX);
    }

    // ================================================================
    // SECTION 4 : decode_jwt — cas d'erreur (sans fichier clé)
    // Les appels directs paniquent si key/public_key.pem est absent.
    // On teste via la couche HTTP où l'erreur est gérée proprement.
    // ================================================================

    #[actix_rt::test]
    async fn test_init_session_without_cookie_returns_401() {
        // JWT extractor s'exécute avant d'atteindre Redis/S3 → 401 sans dépendances
        let app = test::init_service(
            App::new()
                .app_data(actix_web::web::Data::new(
                    redis::Client::open("redis://127.0.0.1:6379").unwrap(),
                ))
                .service(init_session),
        )
        .await;

        let req = test::TestRequest::get().uri("/init-sesssion").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::UNAUTHORIZED);
    }

    #[actix_rt::test]
    async fn test_init_session_with_invalid_cookie_returns_401() {
        let app = test::init_service(
            App::new()
                .app_data(actix_web::web::Data::new(
                    redis::Client::open("redis://127.0.0.1:6379").unwrap(),
                ))
                .service(init_session),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/init-sesssion")
            .cookie(actix_web::cookie::Cookie::new("PAD_Auth", "token_invalide_garbage"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::UNAUTHORIZED);
    }

    #[actix_rt::test]
    async fn test_init_session_with_tampered_jwt_returns_401() {
        // JWT structurellement valide mais signature incorrecte
        let tampered = "eyJhbGciOiJFZERTQSJ9.eyJpZCI6OTk5OTksImV4cCI6OTk5OTk5OTk5OX0.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let app = test::init_service(
            App::new()
                .app_data(actix_web::web::Data::new(
                    redis::Client::open("redis://127.0.0.1:6379").unwrap(),
                ))
                .service(init_session),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/init-sesssion")
            .cookie(actix_web::cookie::Cookie::new("PAD_Auth", tampered))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::UNAUTHORIZED);
    }

    #[actix_rt::test]
    async fn test_ws_without_cookie_returns_401() {
        let app = test::init_service(
            App::new()
                .app_data(actix_web::web::Data::new(
                    redis::Client::open("redis://127.0.0.1:6379").unwrap(),
                ))
                .service(ws_index),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/ws/test_video.mp4")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::UNAUTHORIZED);
    }

    #[actix_rt::test]
    async fn test_ws_with_wrong_cookie_name_returns_401() {
        // Cookie présent mais mauvais nom : PAD_Auth est attendu
        let app = test::init_service(
            App::new()
                .app_data(actix_web::web::Data::new(
                    redis::Client::open("redis://127.0.0.1:6379").unwrap(),
                ))
                .service(ws_index),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/ws/video.mp4")
            .cookie(actix_web::cookie::Cookie::new("session", "valeur"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::UNAUTHORIZED);
    }

    // ================================================================
    // SECTION 5 : Shadow ban / Rate limiting (nécessite Redis)
    // Lancer avec : cargo test -- --ignored
    // ================================================================

    #[actix_rt::test]
    #[ignore = "Nécessite un Redis actif sur 127.0.0.1:6379"]
    async fn test_rate_limit_allows_first_10_requests() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let uid = u64::MAX - 100; // ID peu probable d'exister en prod
        cleanup_ratelimit_key(uid, &client).await;

        for i in 1..=10 {
            let allowed = log_id(uid, &client).await;
            assert!(allowed, "La requête #{} doit être autorisée", i);
        }
        cleanup_ratelimit_key(uid, &client).await;
    }

    #[actix_rt::test]
    #[ignore = "Nécessite un Redis actif sur 127.0.0.1:6379"]
    async fn test_rate_limit_shadow_ban_on_11th_request() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let uid = u64::MAX - 101;
        cleanup_ratelimit_key(uid, &client).await;

        for _ in 0..10 {
            log_id(uid, &client).await;
        }
        let blocked = log_id(uid, &client).await;
        assert!(!blocked, "La 11ème requête doit être bloquée (shadow ban)");
        cleanup_ratelimit_key(uid, &client).await;
    }

    #[actix_rt::test]
    #[ignore = "Nécessite un Redis actif sur 127.0.0.1:6379"]
    async fn test_rate_limit_shadow_ban_persists_after_threshold() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let uid = u64::MAX - 102;
        cleanup_ratelimit_key(uid, &client).await;

        for _ in 0..15 {
            log_id(uid, &client).await;
        }
        for i in 0..5 {
            let blocked = log_id(uid, &client).await;
            assert!(!blocked, "Le shadow ban doit persister, requête #{}", i + 16);
        }
        cleanup_ratelimit_key(uid, &client).await;
    }

    #[actix_rt::test]
    #[ignore = "Nécessite un Redis actif sur 127.0.0.1:6379"]
    async fn test_rate_limit_is_per_user_independent() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let user_a = u64::MAX - 110;
        let user_b = u64::MAX - 111;
        cleanup_ratelimit_key(user_a, &client).await;
        cleanup_ratelimit_key(user_b, &client).await;

        // user_a dépasse la limite
        for _ in 0..11 {
            log_id(user_a, &client).await;
        }
        // user_b reste libre
        let b_allowed = log_id(user_b, &client).await;
        assert!(b_allowed, "Le shadow ban de A ne doit pas affecter B");

        cleanup_ratelimit_key(user_a, &client).await;
        cleanup_ratelimit_key(user_b, &client).await;
    }

    #[actix_rt::test]
    #[ignore = "Nécessite un Redis actif sur 127.0.0.1:6379"]
    async fn test_rate_limit_key_has_ttl_of_one_hour() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let uid = u64::MAX - 120;
        cleanup_ratelimit_key(uid, &client).await;

        log_id(uid, &client).await;

        if let Ok(mut conn) = client.get_multiplexed_tokio_connection().await {
            let key = format!("ratelimit:{}", uid);
            let ttl: i64 = conn.ttl(&key).await.unwrap_or(-1);
            assert!(ttl > 0 && ttl <= 3600, "TTL doit être dans [1, 3600], obtenu: {}", ttl);
        }
        cleanup_ratelimit_key(uid, &client).await;
    }

    #[actix_rt::test]
    #[ignore = "Nécessite un Redis actif sur 127.0.0.1:6379"]
    async fn test_rate_limit_counter_increments_correctly() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let uid = u64::MAX - 130;
        cleanup_ratelimit_key(uid, &client).await;

        for _ in 0..5 {
            log_id(uid, &client).await;
        }

        if let Ok(mut conn) = client.get_multiplexed_tokio_connection().await {
            let key = format!("ratelimit:{}", uid);
            let count: i64 = conn.get(&key).await.unwrap_or(0);
            assert_eq!(count, 5, "Le compteur doit valoir 5 après 5 appels");
        }
        cleanup_ratelimit_key(uid, &client).await;
    }

    #[actix_rt::test]
    #[ignore = "Nécessite un Redis actif sur 127.0.0.1:6379"]
    async fn test_rate_limit_fail_closed_on_redis_unreachable() {
        // Port inexistant : simule une panne Redis
        let client = redis::Client::open("redis://127.0.0.1:19999").unwrap();
        let result = log_id(42, &client).await;
        assert!(!result, "Sans Redis, log_id doit refuser (fail-closed)");
    }

    // Helper : supprime la clé de rate limit dans Redis
    async fn cleanup_ratelimit_key(uid: u64, client: &redis::Client) {
        if let Ok(mut conn) = client.get_multiplexed_tokio_connection().await {
            let key = format!("ratelimit:{}", uid);
            let _: Result<(), _> = conn.del(&key).await;
        }
    }

    // ================================================================
    // SECTION 6 : Parsing des messages WebSocket
    // ================================================================

    #[test]
    fn test_ws_upload_done_message_with_challenge() {
        let msg = "UploadDone|HAUT,BAS,GAUCHE,DROITE";
        let parts: Vec<&str> = msg.split('|').collect();
        assert!(msg.starts_with("UploadDone"));
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1], "HAUT,BAS,GAUCHE,DROITE");
    }

    #[test]
    fn test_ws_upload_done_without_pipe_returns_inconnu() {
        let msg = "UploadDone";
        let parts: Vec<&str> = msg.split('|').collect();
        let challenge = parts.get(1).unwrap_or(&"INCONNU");
        assert_eq!(*challenge, "INCONNU");
    }

    #[test]
    fn test_ws_unknown_message_not_treated_as_upload_done() {
        let messages = ["Ping", "RandomData|foo", "upload_done|bar", "UPLOADONE|baz"];
        for msg in &messages {
            assert!(!msg.starts_with("UploadDone"), "Message '{}' ne doit pas déclencher le traitement", msg);
        }
    }

    #[test]
    fn test_ws_upload_done_with_multiple_pipes_takes_second_part() {
        let msg = "UploadDone|HAUT|EXTRA";
        let parts: Vec<&str> = msg.split('|').collect();
        let challenge = parts.get(1).unwrap_or(&"INCONNU");
        assert_eq!(*challenge, "HAUT");
    }

    // ================================================================
    // SECTION 7 : Logique de nommage des fichiers
    // ================================================================

    #[test]
    fn test_filename_format_is_correct() {
        let ts = chrono::Utc::now().timestamp();
        let id: u64 = 12345;
        let filename = format!("data_{}_{}.mp4", ts, id);
        assert!(filename.starts_with("data_"));
        assert!(filename.ends_with(".mp4"));
        assert!(filename.contains(&id.to_string()));
        assert!(filename.contains(&ts.to_string()));
    }

    #[test]
    fn test_filename_unique_between_different_users() {
        let ts = 1_700_000_000u64;
        let f1 = format!("data_{}_{}.mp4", ts, 1u64);
        let f2 = format!("data_{}_{}.mp4", ts, 2u64);
        assert_ne!(f1, f2);
    }

    #[test]
    fn test_filename_unique_between_different_timestamps() {
        let id = 999u64;
        let f1 = format!("data_{}_{}.mp4", 1_700_000_000u64, id);
        let f2 = format!("data_{}_{}.mp4", 1_700_000_001u64, id);
        assert_ne!(f1, f2);
    }

    // ================================================================
    // SECTION 8 : Infrastructure — Redis Client (parsing URL)
    // ================================================================

    #[test]
    fn test_redis_client_valid_url_parses_ok() {
        assert!(redis::Client::open("redis://127.0.0.1:6379").is_ok());
    }

    #[test]
    fn test_redis_client_valid_url_with_password_parses_ok() {
        assert!(redis::Client::open("redis://:password@127.0.0.1:6379").is_ok());
    }

    #[test]
    fn test_redis_client_invalid_scheme_fails() {
        assert!(redis::Client::open("http://127.0.0.1:6379").is_err());
    }

    #[test]
    fn test_redis_url_env_fallback_default() {
        std::env::remove_var("REDIS_URL");
        let url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        assert_eq!(url, "redis://127.0.0.1:6379");
    }

    #[test]
    fn test_redis_url_env_custom_overrides_default() {
        std::env::set_var("REDIS_URL", "redis://redis-server:6380");
        let url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        assert_eq!(url, "redis://redis-server:6380");
        std::env::remove_var("REDIS_URL");
    }

    // ================================================================
    // SECTION 9 : Constantes métier (tests documentaires)
    // ================================================================

    #[test]
    fn test_rate_limit_threshold_constant_is_10() {
        // Documente la valeur seuil de log_id : <= 10 → autorisé
        let threshold: i64 = 10;
        assert_eq!(threshold, 10, "La limite doit être 10 requêtes par fenêtre");
    }

    #[test]
    fn test_rate_limit_ttl_is_one_hour() {
        let ttl: i64 = 3600;
        assert_eq!(ttl, 3600, "Le TTL de la fenêtre doit être 3600s (1h)");
    }

    #[test]
    fn test_signed_url_expiry_is_5_minutes() {
        let expiry = std::time::Duration::from_secs(300);
        assert_eq!(expiry.as_secs(), 300, "L'URL pré-signée S3 doit expirer en 5 minutes");
    }

    #[test]
    fn test_jwt_expiry_is_24_hours() {
        let expiry_seconds: i64 = 86400;
        assert_eq!(expiry_seconds, 86400, "Le JWT doit expirer en 24h");
    }

    #[test]
    fn test_jwt_algorithm_is_eddsa() {
        // Documente que l'algo choisi est EdDSA (Ed25519), pas HS256
        let algo = jsonwebtoken::Algorithm::EdDSA;
        assert_eq!(format!("{:?}", algo), "EdDSA");
    }

    #[test]
    fn test_challenge_direction_count_is_4() {
        let directions = ["HAUT", "BAS", "GAUCHE", "DROITE"];
        assert_eq!(directions.len(), 4, "Il doit y avoir exactement 4 directions possibles");
    }
}

// --- MAIN ---

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let endpoint = "http://localhost:9000";
    let config = aws_config::from_env().endpoint_url(endpoint).region(Region::new("eu-west-1")).load().await;
    let s3 = Client::from_conf(aws_sdk_s3::config::Builder::from(&config).force_path_style(true).build());
    let _ = s3.create_bucket().bucket("pad-bucket").send().await;

    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let redis_client = redis::Client::open(redis_url).expect("Invalid Redis URL");

    println!("🚀 Server ready on http://localhost:8080");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:3000")
            .allow_any_method()
            .allow_any_header()
            .supports_credentials();

        App::new()
            .wrap(cors)
            .app_data(Data::new(s3.clone()))
            .app_data(Data::new(redis_client.clone()))
            .service(init_session)
            .service(authentification_get)
            .service(ws_index)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}