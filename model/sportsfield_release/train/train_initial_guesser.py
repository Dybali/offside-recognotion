"""
train the initial guesser on homography data
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.abspath(os.getcwd()))
from torch.utils.data import DataLoader
from models import init_guesser, end_2_end_optimization_helper
from options import options
from datasets import aligned_dataset
from utils import metrics, utils, warp
import torch.nn as nn
from torchsummary import summary
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import torchvision.transforms as transforms
import random

class FootballFieldAugmentation:
    def __init__(self, crop_size=(640, 640)):
        self.crop_size = crop_size
        
    def __call__(self, frame, homography):
        # Random crop ve resize
        if random.random() < 0.5:
            h, w = frame.shape[1:]
            crop_h = random.randint(int(h*0.8), h)
            crop_w = random.randint(int(w*0.8), w)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            frame = frame[:, top:top+crop_h, left:left+crop_w]
            frame = transforms.Resize(self.crop_size)(frame)
            
            # Homography matrisini güncelle
            scale_x = self.crop_size[0] / crop_w
            scale_y = self.crop_size[1] / crop_h
            translation = torch.tensor([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=homography.dtype)
            scaling = torch.tensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=homography.dtype)
            homography = torch.matmul(scaling, torch.matmul(translation, homography))
        
        # Random horizontal flip
        if random.random() < 0.5:
            frame = transforms.RandomHorizontalFlip(p=1.0)(frame)
            # Homography matrisini güncelle
            flip_matrix = torch.tensor([[-1, 0, frame.shape[2]], [0, 1, 0], [0, 0, 1]], dtype=homography.dtype)
            homography = torch.matmul(flip_matrix, homography)
        
        # Random brightness/contrast
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            frame = transforms.ColorJitter(brightness=brightness, contrast=contrast)(frame)
        
        # Random rotation (küçük açılarla)
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            frame = transforms.RandomRotation([angle, angle])(frame)
            # Homography matrisini güncelle
            rad = np.radians(angle)
            cos_theta = np.cos(rad)
            sin_theta = np.sin(rad)
            center_x = frame.shape[2] / 2
            center_y = frame.shape[1] / 2
            rotation_matrix = torch.tensor([
                [cos_theta, -sin_theta, center_x * (1 - cos_theta) + center_y * sin_theta],
                [sin_theta, cos_theta, center_y * (1 - cos_theta) - center_x * sin_theta],
                [0, 0, 1]
            ], dtype=homography.dtype)
            homography = torch.matmul(rotation_matrix, homography)
        
        # Random noise
        if random.random() < 0.3:
            noise = torch.randn_like(frame) * 0.05
            frame = torch.clamp(frame + noise, 0, 1)
        
        return frame, homography

def calculate_metrics(original_corners, inferred_corners):
    # Köşe noktaları arasındaki ortalama mesafe
    corner_distances = torch.norm(original_corners - inferred_corners, dim=1)
    mean_distance = torch.mean(corner_distances)
    
    # Maksimum hata
    max_error = torch.max(corner_distances)
    
    # Doğruluk oranı (belirli bir eşik değerine göre)
    threshold = 10.0  # piksel cinsinden
    accuracy = torch.mean((corner_distances < threshold).float())
    
    return {
        'mean_distance': mean_distance.item(),
        'max_error': max_error.item(),
        'accuracy': accuracy.item()
    }

def smooth_loss(losses, window_size=5):
    """Hareketli ortalama ile loss değerlerini yumuşat"""
    smoothed = []
    for i in range(len(losses)):
        start_idx = max(0, i - window_size + 1)
        smoothed.append(np.mean(losses[start_idx:i+1]))
    return smoothed

def visualize_predictions(frame, original_corners, inferred_corners, epoch, save_dir):
    plt.figure(figsize=(10, 10))
    
    # Tensor'ları CPU'ya taşı ve numpy dizilerine dönüştür
    frame = frame.cpu().numpy().transpose(1, 2, 0)
    original_corners = original_corners.cpu().numpy()
    inferred_corners = inferred_corners.cpu().numpy()
    
    # Görüntüyü normalize et
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    
    plt.imshow(frame)
    
    # Gerçek köşeler
    plt.scatter(original_corners[0::2], original_corners[1::2], 
                c='green', label='Gerçek Köşeler')
    
    # Tahmin edilen köşeler
    plt.scatter(inferred_corners[0::2], inferred_corners[1::2], 
                c='red', label='Tahmin Edilen Köşeler')
    
    plt.title(f'Epoch {epoch} - Tahmin Görselleştirmesi')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'prediction_epoch_{epoch}.png'))
    plt.close()

def save_metrics(metrics, epoch, save_path):
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch}:\n")
        f.write(f"Ortalama Mesafe: {metrics['mean_distance']:.2f} piksel\n")
        f.write(f"Maksimum Hata: {metrics['max_error']:.2f} piksel\n")
        f.write(f"Dogruluk Orani: {metrics['accuracy']*100:.2f}%\n\n")

def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for frame, _, gt_homography in test_loader:
            frame = frame.to(device)
            gt_homography = gt_homography.to(device)
            
            inferred_corners = model(frame)
            lower_canon_4pts = end_2_end_optimization_helper.get_default_canon4pts(1, 'lower').to(device)
            original_corners = warp.get_four_corners(gt_homography, lower_canon_4pts[0])
            original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
            
            metrics = calculate_metrics(original_corners, inferred_corners)
            all_metrics.append(metrics)
    
    # Ortalama metrikleri hesapla
    avg_metrics = {
        'mean_distance': np.mean([m['mean_distance'] for m in all_metrics]),
        'max_error': np.mean([m['max_error'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics])
    }
    
    return avg_metrics

def train_batch(frame, homography, base_model, optimizer, loss_fn, batch_size, device):
    base_model.train()
    optimizer.zero_grad()
    inferred_corners = base_model(frame)
    lower_canon_4pts = end_2_end_optimization_helper.get_default_canon4pts(batch_size, canon4pts_type='lower').to(device)
    original_corners = warp.get_four_corners(homography, lower_canon_4pts[0])
    original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
    loss = loss_fn(original_corners, inferred_corners)
    loss.backward()
    optimizer.step()

    # Metrikleri hesapla
    metrics = calculate_metrics(original_corners, inferred_corners)

    return loss, metrics

def validate_batch(frame, homography, base_model, loss_fn, batch_size, device):
    base_model.eval()
    with torch.no_grad():
        inferred_corners = base_model(frame)
    lower_canon_4pts = end_2_end_optimization_helper.get_default_canon4pts(batch_size, canon4pts_type='lower').to(device)
    original_corners = warp.get_four_corners(homography, lower_canon_4pts[0])
    original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
    loss = loss_fn(original_corners, inferred_corners)

    # Metrikleri hesapla
    metrics = calculate_metrics(original_corners, inferred_corners)

    return loss, metrics

def main():
    utils.fix_randomness()
    opt = options.set_init_guesser_train_options()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Batch size'ı artır
    opt.batch_size = 64

    # Epoch sayısını artır
    opt.train_epochs = 100  # 40'tan 100'e çıkardık

    # Metrikler için klasör oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.path.join(opt.out_dir, 'trained_init_guess', f'metrics_{timestamp}')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, 'training_metrics.txt')

    # Data augmentation
    augmentation = FootballFieldAugmentation()

    train_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'train')
    # Train dataset'e augmentation ekle
    train_dataset.transform = augmentation

    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )

    val_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'val')
    val_loader = DataLoader(
        val_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    initial_guesser = init_guesser.InitialGuesserFactory.get_initial_guesser(opt).to(device)
    
    # Dropout ekle
    if hasattr(initial_guesser, 'dropout'):
        initial_guesser.dropout = nn.Dropout(p=0.3)  # %30 dropout
    
    summary(initial_guesser, (3, 640, 640), device=str(device))

    # Learning rate'i düşür
    lr = 5e-5  # 1e-4'ten 5e-5'e düşürüldü
    criterion = nn.SmoothL1Loss()
    optim = torch.optim.Adam(
        params=initial_guesser.parameters(),
        lr=lr,
        weight_decay=1e-4,  # weight decay artırıldı
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler'ı güncelle
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-6
    )

    best_loss = float('inf')
    best_model_state = initial_guesser.state_dict()
    best_optim_state = optim.state_dict()
    best_epoch = 0
    best_iteration = 0
    train_loss, val_loss = [], []
    iteration = 0
    early_stop_counter = 0
    patience = 20  # patience azaltıldı

    # Loss smoothing için pencere boyutu
    window_size = 5

    for epoch in range(opt.train_epochs):
        print(f"Epoch {epoch + 1} out of {opt.train_epochs}")
        epoch_train_loss, epoch_val_loss = [], []
        epoch_train_metrics, epoch_val_metrics = [], []

        # Eğitim
        initial_guesser.train()  # train moduna al
        for _, data_batch in enumerate(train_loader):
            frame, _, gt_homography = data_batch
            frame = frame.to(device)
            gt_homography = gt_homography.to(device)
            loss, metrics = train_batch(frame, gt_homography, initial_guesser, optim, criterion, opt.batch_size, device)
            epoch_train_loss.append(loss.item())
            epoch_train_metrics.append(metrics)
            iteration += 1

        # Validasyon
        initial_guesser.eval()  # eval moduna al
        with torch.no_grad():  # gradient hesaplamayı kapat
            for _, data_batch in enumerate(val_loader):
                frame, _, gt_homography = data_batch
                frame = frame.to(device)
                gt_homography = gt_homography.to(device)
                loss, metrics = validate_batch(frame, gt_homography, initial_guesser, criterion, opt.batch_size, device)
                epoch_val_loss.append(loss.item())
                epoch_val_metrics.append(metrics)

        # Loss smoothing uygula
        smoothed_train_loss = smooth_loss(epoch_train_loss, window_size)
        smoothed_val_loss = smooth_loss(epoch_val_loss, window_size)

        # Metrikleri hesapla ve kaydet
        avg_train_metrics = {
            'mean_distance': np.mean([m['mean_distance'] for m in epoch_train_metrics]),
            'max_error': np.mean([m['max_error'] for m in epoch_train_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in epoch_train_metrics])
        }
        
        avg_val_metrics = {
            'mean_distance': np.mean([m['mean_distance'] for m in epoch_val_metrics]),
            'max_error': np.mean([m['max_error'] for m in epoch_val_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in epoch_val_metrics])
        }

        # Metrikleri kaydet
        save_metrics(avg_train_metrics, epoch + 1, metrics_file)
        save_metrics(avg_val_metrics, epoch + 1, metrics_file)

        # Her 10 epoch'ta bir görselleştirme yap
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                inferred_corners = initial_guesser(frame[0:1])
                lower_canon_4pts = end_2_end_optimization_helper.get_default_canon4pts(1, 'lower').to(device)
                original_corners = warp.get_four_corners(gt_homography[0:1], lower_canon_4pts[0])
                original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
            
            visualize_predictions(frame[0], original_corners[0], inferred_corners[0], 
                                epoch + 1, metrics_dir)

        train_loss.append(np.mean(smoothed_train_loss))
        val_loss.append(np.mean(smoothed_val_loss))
        print(f"Training loss: {train_loss[-1]:.4f}")
        print(f"Validation loss: {val_loss[-1]:.4f}")
        print(f"Training metrics: {avg_train_metrics}")
        print(f"Validation metrics: {avg_val_metrics}")

        scheduler.step(val_loss[-1])

        if val_loss[-1] < best_loss:
            best_model_state = initial_guesser.state_dict()
            best_optim_state = optim.state_dict()
            best_loss = val_loss[-1]
            best_epoch = epoch + 1
            best_iteration = iteration
            early_stop_counter = 0
            print(f" YENİ EN İYİ MODEL! Epoch {best_epoch}, Validation Loss: {best_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f" Erken durdurma sayacı: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(" Erken durdurma tetiklendi.")
                break

        epoch_ckpt = os.path.join(opt.out_dir, 'trained_init_guess', f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(initial_guesser.state_dict(), epoch_ckpt)

    # Test seti üzerinde değerlendirme
    test_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=0,  
        pin_memory=True  # CUDA için 
    )
    test_metrics = evaluate_on_test_set(initial_guesser, test_loader, device)
    
    print("\nTest Seti Metrikleri:")
    print(f"Ortalama Mesafe: {test_metrics['mean_distance']:.2f} piksel")
    print(f"Maksimum Hata: {test_metrics['max_error']:.2f} piksel")
    print(f"Doğruluk Oranı: {test_metrics['accuracy']*100:.2f}%")

    final_ckpt = {
        'epoch': best_epoch,
        'iteration': best_iteration,
        'optim_state_dict': best_optim_state,
        'model_state_dict': best_model_state,
        'best_val_loss': best_loss,
        'test_metrics': test_metrics
    }
    final_out_path = os.path.join(opt.out_dir, 'trained_init_guess', 'final_best_model.pth.tar')
    torch.save(final_ckpt, final_out_path)
    print(f"\n Eğitim tamamlandı! En iyi model Epoch {best_epoch}'de kaydedildi (Validation Loss: {best_loss:.4f})")

    # Loss grafiği - smoothed
    epochs = np.arange(len(train_loss)) + 1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, np.float32(train_loss), 'bo', label='Training loss (smoothed)')
    plt.plot(epochs, np.float32(val_loss), 'r', label='Validation loss (smoothed)')
    plt.title('Training and Validation loss over epochs (Smoothed)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, 'loss_fig.png'))
    plt.close()

    # Metrik grafikleri - smoothed
    plt.figure(figsize=(15, 5))
    
    # Train ve validation metriklerini ayrı ayrı topla
    train_mean_distances = [m['mean_distance'] for m in epoch_train_metrics]
    val_mean_distances = [m['mean_distance'] for m in epoch_val_metrics]
    train_max_errors = [m['max_error'] for m in epoch_train_metrics]
    val_max_errors = [m['max_error'] for m in epoch_val_metrics]
    train_accuracies = [m['accuracy']*100 for m in epoch_train_metrics]
    val_accuracies = [m['accuracy']*100 for m in epoch_val_metrics]
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, smooth_loss(train_mean_distances, window_size), 'bo', label='Training')
    plt.plot(epochs, smooth_loss(val_mean_distances, window_size), 'r', label='Validation')
    plt.title('Ortalama Mesafe (Smoothed)')
    plt.xlabel('Epochs')
    plt.ylabel('Piksel')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, smooth_loss(train_max_errors, window_size), 'bo', label='Training')
    plt.plot(epochs, smooth_loss(val_max_errors, window_size), 'r', label='Validation')
    plt.title('Maksimum Hata (Smoothed)')
    plt.xlabel('Epochs')
    plt.ylabel('Piksel')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, smooth_loss(train_accuracies, window_size), 'bo', label='Training')
    plt.plot(epochs, smooth_loss(val_accuracies, window_size), 'r', label='Validation')
    plt.title('Doğruluk Oranı (Smoothed)')
    plt.xlabel('Epochs')
    plt.ylabel('Yüzde (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'metrics_fig.png'))
    plt.close()

if __name__ == '__main__':
    main()
