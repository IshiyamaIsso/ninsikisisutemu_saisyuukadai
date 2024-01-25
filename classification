import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# データセットのパス
dataset_path = "dataset"

# データ拡張と正規化
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# データセットの読み込み
dataset = datasets.ImageFolder(dataset_path, transform=transform)

# データセットを訓練用と検証用に分割
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)


# モデルの構築
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 128 * 3, 128)
        self.relu = nn.Softmax()
        self.fc2 = nn.Linear(128, 5)  # クラス数に合わせて調整
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleModel()

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ベストモデルの初期化


# モデルの訓練
num_epochs = 20
def main():
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # 検証
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}')

        # ベストモデルの保存
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best Val Accuracy: {best_val_accuracy:.4f}')

if __name__ =="__main__":
    main()
    '''
    # 最終的な訓練精度と検証精度の表示
    final_train_loss, final_train_accuracy = train_loss, train_accuracy
    final_val_loss, final_val_accuracy = val_loss, val_accuracy


    # ベストモデルの読み込みと表示
    best_model = SimpleModel()
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.eval()

    # テストデータセットの読み込みと評価
    test_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = best_model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            


    print(f'Best Model Test Accuracy: {best_val_accuracy:.4f}')

if __name__ =="__main__":
    main()'''
