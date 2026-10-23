"""Reviewed visual/text encoders for the frozen A4500 experiment."""

CLIP_ENCODER = "openai/clip-vit-base-patch32"
SIGLIP2_ENCODER = "google/siglip2-base-patch16-224"
A4500_ENCODERS = (CLIP_ENCODER, SIGLIP2_ENCODER)


def encoder_family(model_name: str) -> str:
    """Resolve the reviewed SigLIP2 model or official OpenAI CLIP family."""
    # Retain the general adapter's support for the other official CLIP sizes;
    # production A4500 protocols still use the exact allowlist above.
    if model_name.startswith("openai/clip-"):
        return "clip"
    if model_name == SIGLIP2_ENCODER:
        return "siglip2"
    raise ValueError(
        f"Unsupported encoder {model_name!r}; choose one of {A4500_ENCODERS}"
    )
