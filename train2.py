import torch
from ultralytics import YOLO

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def train_lung_cancer_improved():
    device = 'cuda:0'

    model = YOLO('yolov8s-seg.pt')

    train_args = dict(
        data='data.yaml',
        epochs=120,
        imgsz=640,
        batch=12,
        device=device,
        workers=2,
        optimizer='AdamW',
        patience=40,
        save=True,
        plots=True,
        close_mosaic=20,

        amp=False,
        lr0=0.001,

        box=7.5,
        cls=1.5,
        dfl=1.5,

        mosaic=0.3,
        mixup=0.1,
        copy_paste=0.2,

        cache=False,
        seed=42,
    )

    results = model.train(**train_args)

    return results


if __name__ == '__main__':
    train_lung_cancer_improved()