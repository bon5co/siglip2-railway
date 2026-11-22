# siglip2-railway

SigLIP 2 image and text embedding API with zero-shot image classification for [Railway](https://railway.com), CPU only. Serves [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) (Apache-2.0 per its model card) behind one authenticated HTTP API. Images and text land in the same 768-dimensional space, so the vectors work for text-to-image and image-to-image search in any vector database.

- `POST /embed/image` with multipart `file` parts (up to 16) or one raw image body: one L2-normalised vector per image (`normalize=false` for raw).
- `POST /v1/embeddings` (alias `/embed/text`) with JSON `{"input": "text" | ["text", ...]}`: OpenAI embeddings response shape, 64 tokens per text.
- `POST /classify` with one image and `labels` (comma-separated or JSON array, query or form field), optional `template` (default `This is a photo of {}.`): a sigmoid `probability` and a `softmax` per label, best first.
- `GET /health` (503 until the model is loaded and warm, then 200), `GET /v1/models`.
- Auth: `Authorization: Bearer $API_KEY` on every route except `/health`. An empty `API_KEY` refuses to start.
- Env: see `env.md` in the template repository; `API_KEY` is the only required one.

The weights are baked into the image, pinned to one Hugging Face commit and verified by sha256 at build. Images and text are held in memory for one request and never logged or stored. Wrapper code is Apache-2.0 (`LICENSE`); see `NOTICE`.

Image: `ghcr.io/bon5co/siglip2-railway`.
