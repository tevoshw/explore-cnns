# Architectures of famous CNNs



# 1. Image Classification

## 1.1 VGG (Winner of ImageNet 2014)

É o clássico, com blocos pequenos de matrizes (3x3)



```


VGG - Exemplo Sequência
Input → Output Cada Bloco
Input: (1, 3, 224, 224)
       batch=1, canais=3, altura=224, largura=224

BLOCO 1:
├─ Conv 3×3 (64 canais, padding=1) → (1, 64, 224, 224)
├─ Conv 3×3 (64 canais, padding=1) → (1, 64, 224, 224)
└─ MaxPool 2×2 → (1, 64, 112, 112)  ← altura/largura ÷2

BLOCO 2:
├─ Conv 3×3 (128 canais) → (1, 128, 112, 112)
├─ Conv 3×3 (128 canais) → (1, 128, 112, 112)
└─ MaxPool 2×2 → (1, 128, 56, 56)   ← altura/largura ÷2

BLOCO 3:
├─ Conv 3×3 (256 canais) → (1, 256, 56, 56)
├─ Conv 3×3 (256 canais) → (1, 256, 56, 56)
├─ Conv 3×3 (256 canais) → (1, 256, 56, 56)
└─ MaxPool 2×2 → (1, 256, 28, 28)   ← altura/largura ÷2

BLOCO 4:
├─ Conv 3×3 (512 canais) → (1, 512, 28, 28)
├─ Conv 3×3 (512 canais) → (1, 512, 28, 28)
├─ Conv 3×3 (512 canais) → (1, 512, 28, 28)
└─ MaxPool 2×2 → (1, 512, 14, 14)   ← altura/largura ÷2

BLOCO 5:
├─ Conv 3×3 (512 canais) → (1, 512, 14, 14)
├─ Conv 3×3 (512 canais) → (1, 512, 14, 14)
├─ Conv 3×3 (512 canais) → (1, 512, 14, 14)
└─ MaxPool 2×2 → (1, 512, 7, 7)     ← altura/largura ÷2

AdaptiveAvgPool → (1, 512, 7, 7)

Flatten → (1, 25088)

Dense(4096) → (1, 4096)
Dense(4096) → (1, 4096)
Dense(1000) → (1, 1000)  ← classificação


```


## 1.2 ResNet (Winner of ImageNet 2015)



```

Input (224×224×3)
  ↓
Conv 7×7 + MaxPool → (56×56×64)
  ↓
BLOCO 1 (64 canais):
* 3 ResidualBlocks
  ├─ Conv 1×1 → Conv 3×3 → Conv 1×1 = F(x)
  ├─ Shortcut(x) = x (identidade)
  └─ Output = F(x) + x
  ├─ Conv 1×1 → Conv 3×3 → Conv 1×1 = F(x)
  ├─ Shortcut(x) = x
  └─ Output = F(x) + x
  ├─ Conv 1×1 → Conv 3×3 → Conv 1×1 = F(x)
  ├─ Shortcut(x) = x
  └─ Output = F(x) + x
  ↓
BLOCO 2 (256 canais, stride=2):
* 4 ResidualBlocks
  ├─ Conv 1×1 → Conv 3×3 → Conv 1×1 = F(x)
  ├─ Shortcut(x) = Conv 1×1 stride=2 (adapta)
  └─ Output = F(x) + Shortcut(x)
  └─ (próximos 3: F(x) + x)
  ↓
BLOCO 3 (512 canais, stride=2):
* 6 ResidualBlocks (primeiro Com shortcut Conv, resto identidade)
  ↓
BLOCO 4 (2048 canais, stride=2):
* 3 ResidualBlocks (primeiro Com shortcut Conv, resto identidade)
  ↓
AvgPool → Dense(1000)


```




## 1.3 Efficient Net 



## 1.4 Vision Transformers (ViT)