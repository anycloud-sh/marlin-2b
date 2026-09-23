FROM pytorch/pytorch@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

ARG IMAGE_REVISION
LABEL org.opencontainers.image.title="Marlin-2B video captions and event localization" \
      org.opencontainers.image.description="AnyCloud-powered finite GPU inference using the pinned creator helpers" \
      org.opencontainers.image.source="https://github.com/anycloud-sh/marlin-2b" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-3.0" \
      sh.anycloud.upstream.model="NemoStation/Marlin-2B" \
      sh.anycloud.upstream.model-revision="fd111fca4fc7897876fb0d7e9df22ca5ac8ab965"

WORKDIR /opt/marlin
COPY runtime_setup.py ./
RUN python3 runtime_setup.py && rm -rf /var/lib/apt/lists/* /root/.cache/pip
COPY infer.py LICENSE NOTICE ./
COPY examples ./examples
ENV PATH=/opt/marlin-runtime/bin:$PATH \
    PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
ENTRYPOINT ["/opt/marlin-runtime/bin/python", "/opt/marlin/infer.py"]
