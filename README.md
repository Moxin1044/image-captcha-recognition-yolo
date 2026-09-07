# Image CAPTCHA Recognition with YOLO

基于 Ultralytics YOLO 分类模型的固定宽度图片验证码识别项目。模型先将验证码按字符位置等宽切分，再识别每个字符，最后从左到右拼接结果。

## 当前模型

当前训练好的字符分类模型已发布到 [GitHub Release V1.0.0](https://github.com/Moxin1044/image-captcha-recognition-yolo/releases/tag/V1.0.0)：

- [直接下载 `captcha-character-classifier-yolo11n-100e.pt`](https://github.com/Moxin1044/image-captcha-recognition-yolo/releases/download/V1.0.0/captcha-character-classifier-yolo11n-100e.pt)
- 模型：YOLO11n-cls（预训练权重微调）
- 训练轮数：100（最佳轮次：94）
- 输入尺寸：96×96
- 字符类别：36 类（数字 `0-9`，大写字母 `A-Z`）
- 验证集单字符 Top-1：83.23%
- 验证集单字符 Top-5：94.59%

这里的准确率是“单字符准确率”。整串验证码必须每个字符都正确，因此 4 位或 5 位整串准确率会更低。

## 安装

```bash
python -m pip install -r requirements.txt
```

## 数据准备

### 使用已有清单

`Verification_code/train.txt` 和 `Verification_code/val.txt` 每行格式为：

```text
图片文件名 验证码文本
```

按清单准备字符裁剪数据：

```bash
python prepare_dataset.py --source Verification_code --output data/characters --clear
```

### 直接使用图片文件名作为标签

如果图片文件名就是验证码文本（例如 `000a.png` 或 `KYHXV.png`），可以自动发现标签并按固定哈希划分训练集/验证集。该方式会同时支持 4 位和 5 位验证码：

```bash
python prepare_dataset.py --source Verification_code --output data/characters --discover --clear
```

脚本不会修改原始图片，只会在 `data/characters/train/<字符>/` 和 `data/characters/val/<字符>/` 下生成字符裁剪图。

## 训练

GPU：

```bash
python train.py --data data/characters --epochs 100 --imgsz 96 --device 0
```

CPU：

```bash
python train.py --data data/characters --epochs 100 --imgsz 96 --device cpu
```

训练结果默认写入：

```text
runs/captcha/character_classifier/weights/best.pt
```

训练脚本已关闭左右翻转增强，因为镜像会改变字符形状；同时使用较小的旋转、平移和擦除增强来提升泛化能力。

## 预测

四位验证码：

```bash
python predict.py Verification_code/000a.png --model path/to/best.pt --positions 4
```

五位验证码：

```bash
python predict.py Verification_code/KYHXV.png --model path/to/best.pt --positions 5
```

如果省略 `--model`，脚本会自动查找 `runs/` 下最新的 `weights/best.pt`。输出格式为：

```text
预测文本    confidence=平均字符置信度
```

预测脚本假设字符等宽，并要求图片宽度能被 `--positions` 整除。如果字符旋转严重、相互重叠或不是等宽布局，应改用带边界框标注的 YOLO 检测方案。

## 示例

`examples/` 中包含 10 张示例验证码图片，可用于快速测试：

```text
KYHXV.png  6YNE2.png  Y3FV8.png  CHMHX.png  CSZQX.png
5ZXN3.png  C38HM.png  FWM48.png  ZNYXU.png  ZRZQ6.png
```

## 目录结构

```text
.
├── Verification_code/        # 原始验证码图片和可选清单
├── examples/                 # 示例图片
├── data/characters/          # 生成的字符分类数据集
├── prepare_dataset.py        # 数据准备和字符裁剪
├── train.py                  # YOLO 分类训练
├── predict.py                # 固定位数验证码预测
└── requirements.txt
```

## 许可证

代码和模型仅供学习与研究使用。使用验证码数据或自动识别服务时，请遵守数据来源网站的服务条款和适用法律法规。
