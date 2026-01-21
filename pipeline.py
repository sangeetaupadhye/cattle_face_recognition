def process_face_verification(record):
    """
    Processes a single cattle face verification request end-to-end.
    API, storage, and production integrations removed for confidentiality.
    """

    # Validate input images
    if not record.get("image_1") or not record.get("image_2"):
        return {
            "status": "failed",
            "reason": "missing_images"
        }

    image_1_path = record["image_1"]
    image_2_path = record["image_2"]

    # Preprocess images
    img1_tensor = preprocess_image(image_1_path)
    img2_tensor = preprocess_image(image_2_path)

    # Extract embeddings (ViT + MagFace)
    embedding_1 = model(img1_tensor)
    embedding_2 = model(img2_tensor)

    #  Normalize embeddings (handled in model, shown for clarity)
    # embedding_1 = normalize(embedding_1)
    # embedding_2 = normalize(embedding_2)

    #  Compute similarity
    similarity_score = cosine_similarity(embedding_1, embedding_2)

    #  Identity decision
    is_same_cattle = similarity_score > 0.70

    # Return result
    return {
        "status": "success",
        "similarity_score": round(similarity_score, 4),
        "match": is_same_cattle,
        "decision": "same_cattle" if is_same_cattle else "different_cattle"
    }
