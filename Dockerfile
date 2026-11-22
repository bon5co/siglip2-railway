# SigLIP 2 image and text embedding API with zero-shot classification, Bearer API_KEY.
# Source: https://github.com/bon5co/siglip2-railway
# CPU only (torch), weights baked in (about 1.5 GB), stateless (no volume needed).
# upstream: pytorch/pytorch torch 2.15.0 and huggingface/transformers 5.24.0 (newest releases, checked 2026-11-22) are pinned in requirements.txt
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(torch|transformers|tokenizers|pillow|safetensors|numpy) "

# Weights, pinned to one Hugging Face commit and verified by sha256 so a changed or replaced file
# fails the build instead of shipping. A 1.5 GB download at boot would sit inside Railway's healthcheck window.
#   google/siglip2-base-patch16-224   Apache-2.0 (model card metadata: license: apache-2.0)
ARG HF=https://huggingface.co/google/siglip2-base-patch16-224/resolve/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
ADD --checksum=sha256:fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7 ${HF}/config.json /models/siglip2/config.json
ADD --checksum=sha256:9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc ${HF}/preprocessor_config.json /models/siglip2/preprocessor_config.json
ADD --checksum=sha256:baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351 ${HF}/special_tokens_map.json /models/siglip2/special_tokens_map.json
ADD --checksum=sha256:cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322 ${HF}/tokenizer.json /models/siglip2/tokenizer.json
ADD --checksum=sha256:61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2 ${HF}/tokenizer.model /models/siglip2/tokenizer.model
ADD --checksum=sha256:14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8 ${HF}/tokenizer_config.json /models/siglip2/tokenizer_config.json
ADD --checksum=sha256:612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b ${HF}/model.safetensors /models/siglip2/model.safetensors
ADD --checksum=sha256:39ac3705d62af9ffa1a14675b8ccb220a75f2d81acd530e564a3b1e3dfe418d8 ${HF}/README.md /models/siglip2/MODEL_CARD.md

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin siglip \
    && chmod -R a+rX /models
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh
ENV MODEL_DIR=/models/siglip2
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
