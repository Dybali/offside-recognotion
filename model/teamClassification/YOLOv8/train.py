from ultralytics import YOLO
import torch

# YOLOv8x modelini sıfırdan yükle
model = YOLO('yolov8x.yaml')  # .pt yerine .yaml kullanarak 

# Eğitimi başlat
results = model.train(
    data='dataset/data.yaml',
    epochs=100,                # Epoch sayısı
    imgsz=640,                # Görüntü boyutu
    batch=4,                  # Batch size
    device='0' if torch.cuda.is_available() else 'cpu',
    patience=20,              # Early stopping için 20 epoch bekle
    lr0=0.0001,              # Başlangıç learning rate
    lrf=0.01,                # Final learning rate
    warmup_epochs=5,         # Isınma epoch'ları
    warmup_momentum=0.8,     # Isınma momentumu
    warmup_bias_lr=0.1,      # Isınma bias learning rate
    box=7.5,                 # Box loss gain
    cls=0.5,                 # Class loss gain
    dfl=1.5,                 # DFL loss gain
    close_mosaic=10,         # Mosaic augmentasyonu kapatma epoch'u
    hsv_h=0.015,            # HSV-Hue augmentasyonu
    hsv_s=0.7,              # HSV-Saturation augmentasyonu
    hsv_v=0.4,              # HSV-Value augmentasyonu
    degrees=0.1,            # Rotasyon augmentasyonu
    translate=0.2,          # Öteleme augmentasyonu
    scale=0.7,              # Ölçeklendirme augmentasyonu
    shear=0.0,              # Kesme augmentasyonu
    perspective=0.0,         # Perspektif augmentasyonu
    flipud=0.0,             # Dikey çevirme augmentasyonu
    fliplr=0.5,             # Yatay çevirme augmentasyonu
    mosaic=1.0,             # Mosaic augmentasyonu
    mixup=0.0,              # Mixup augmentasyonu
    copy_paste=0.0          # Kopyala-yapıştır augmentasyonu
)

# Eğitim sonuçlarını yazdır
print("\nEğitim tamamlandı!")
print(f"En iyi mAP: {results.best_map}")
print(f"En iyi epoch: {results.best_epoch}")
print(f"Eğitim süresi: {results.train_time:.2f} saniye")

# Terminalda eğitim için alternatif komut
"""
yolo train model=yolov8x.yaml data=dataset/data.yaml epochs=100 imgsz=640 batch=4 device=0 patience=20 lr0=0.0001 lrf=0.01 warmup_epochs=5 warmup_momentum=0.8 warmup_bias_lr=0.1 box=7.5 cls=0.5 dfl=1.5 close_mosaic=10 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=0.1 translate=0.2 scale=0.7 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 mosaic=1.0 mixup=0.0 copy_paste=0.0
"""