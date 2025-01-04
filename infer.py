import torch
import argparse
import torchvision.transforms as T
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path", type=str, required=True, help="Путь к X-ray изображению."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Путь к файлу .ckpt или .pt модели.",
    )
    args = parser.parse_args()

    model = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    model.eval()

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    image = Image.open(args.image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # batch dimension

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    print(f"Предсказанный класс: {pred_class}, уверенность: {confidence:.4f}")


if __name__ == "__main__":
    main()
