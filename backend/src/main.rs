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
            ctx.text(format!("{}: {}", item.status, item.filename));
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

// --- MAIN ---

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let endpoint = "http://minio:9000";
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