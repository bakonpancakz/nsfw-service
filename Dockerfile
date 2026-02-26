# Prepare Program
FROM golang:1.26-trixie AS build
WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install -y gcc wget tar
RUN rm -rf /var/lib/apt/lists/*

# Using the GPU version although this is the CPU Dockerfile as it uses AVX2 extensions.
# If your machine doesn't support these use this archive URL instead:
# https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-cpu-linux-x86_64.tar.gz

RUN wget -O tensorflow.tar.gz -q https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-gpu-linux-x86_64.tar.gz && \
    tar -C /usr/local -xzf tensorflow.tar.gz && \
    rm tensorflow.tar.gz

ENV CGO_ENABLED=1 \
    CGO_CFLAGS="-I/usr/local/include" \
    CGO_LDFLAGS="-L/usr/local/lib -ltensorflow"

RUN go build -o service.elf main.go

# Prepare Runtime
FROM debian:trixie-slim AS runtime
WORKDIR /app
RUN mkdir -p /data

COPY --from=build /usr/local/lib    /usr/local/lib
COPY --from=build /app/model        ./model
COPY --from=build /app/service.elf  .

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 9000
CMD ["./service.elf"]

