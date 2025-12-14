from ultralytics import YOLO

# Laad pretrained model
model = YOLO("yolov8m.pt")

# Train
model.train(
    data="dataset/data.yaml",
    epochs=90,              # ~2 uur op A100
    imgsz=640,
    batch=64,                # A100 kan dit makkelijk
    device=0,
    workers=8,

    optimizer="AdamW",
    lr0=0.0015,               # goede start voor fine-tuning
    lrf=0.01,

    cos_lr=True,
    warmup_epochs=3,

    weight_decay=0.0005,
    momentum=0.937,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    fliplr=0.5,

    mosaic=1.0,
    mixup=0.1,

    patience=25,              # early stopping
    save=True,
    save_period=10,

    amp=True,                 # mixed precision → sneller
    cache="ram",              # dataset past makkelijk in RAM
)
