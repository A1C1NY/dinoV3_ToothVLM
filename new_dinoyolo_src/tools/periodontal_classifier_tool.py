"""Function-calling adapter for the periodontal disease classifier."""

from pathlib import Path

from new_dinoyolo_src.infer_classifier_periodontal import load_checkpoint


def classify_periodontal_disease(image_path: str) -> dict:
    """Classify an oral image and return the predicted periodontal category."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    checkpoint = Path(__file__).resolve().parents[2] / "res_checkpoints" / "best_val_acc.pth"
    model, class_names, img_size = load_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.inference_mode():
        logits = model(transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device))
        probabilities = F.softmax(logits, dim=1)[0].cpu().tolist()
    ranked = sorted(zip(class_names, probabilities), key=lambda item: item[1], reverse=True)
    return {
        "status": "success",
        "prediction": ranked[0][0],
        "confidence": ranked[0][1],
        "probabilities": dict(ranked),
    }
