# RAGnarök — all-in-one image: Angular frontend, Rust daemon, embedded UI.
#
#   docker build -t ragnarok .
#   docker run -d -p 7878:7878 -v ragnarok-data:/data \
#       -v /path/to/docs:/docs:ro ragnarok
#
# Then open http://localhost:7878 — the bootstrap admin password is printed
# to the container log on first start (or set it via config.toml, mounted
# to /data/config.toml).

# ---- frontend --------------------------------------------------------------
FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY crates/ragnarok-ui/frontend/package.json ./package.json
RUN npm install --no-audit --no-fund
COPY crates/ragnarok-ui/frontend/ ./
# angular.json outputPath is ../assets relative to /app → /assets
RUN npm run build

# ---- backend ---------------------------------------------------------------
FROM rust:1-bookworm AS backend
WORKDIR /app
# manifests first so dependency downloads survive source changes
COPY Cargo.toml Cargo.lock ./
COPY crates/ragnarok-core/Cargo.toml crates/ragnarok-core/Cargo.toml
COPY crates/ragnarok-crawler/Cargo.toml crates/ragnarok-crawler/Cargo.toml
COPY crates/ragnarok-inference/Cargo.toml crates/ragnarok-inference/Cargo.toml
COPY crates/ragnarok-resource/Cargo.toml crates/ragnarok-resource/Cargo.toml
COPY crates/ragnarok-server/Cargo.toml crates/ragnarok-server/Cargo.toml
COPY crates/ragnarok-store/Cargo.toml crates/ragnarok-store/Cargo.toml
COPY crates/ragnarok-ui/Cargo.toml crates/ragnarok-ui/Cargo.toml
COPY src ./src
COPY crates ./crates
# overwrite any stale/copy-ignored assets with the real production build
RUN rm -rf crates/ragnarok-ui/assets
COPY --from=frontend /assets crates/ragnarok-ui/assets
RUN cargo build --release --bin ragnarok

# ---- runtime ---------------------------------------------------------------
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl tesseract-ocr poppler-utils \
       libgomp1 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
# non-root service user; /data holds ragnarok.db + vectorstore + models
RUN useradd -r -u 1000 -m -d /data ragnarok
COPY --from=backend /app/target/release/ragnarok /usr/local/bin/ragnarok
# ONNX Runtime provider dylibs (libwebgpu_dawn etc.) — $ORIGIN rpath finds
# them next to the binary; absent GPU drivers just fail EP registration
# and the session falls back to CPU.
COPY --from=backend /app/target/release/*.so /usr/local/bin/
USER ragnarok
WORKDIR /data
EXPOSE 7878
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD curl -fsS http://localhost:7878/healthz || exit 1
ENTRYPOINT ["/usr/local/bin/ragnarok"]
CMD ["serve", "--bind", "0.0.0.0:7878", "--data-dir", "/data"]
