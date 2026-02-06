FROM rust:1.88 as builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./

RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -f target/release/deps/mon_projet_ws*

COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app

# On installe OpenSSL (souvent requis par les libs Rust)
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

# On récupère le binaire compilé à l'étape 1
COPY --from=builder /app/target/release/mon_projet_ws .

# On n'oublie pas tes clés !
COPY --from=builder /app/key ./key

# On expose le port 8080
EXPOSE 8080

# La commande pour lancer l'app
CMD ["./mon_projet_ws"]