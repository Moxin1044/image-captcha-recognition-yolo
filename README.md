# Image CAPTCHA Recognition Model (YOLO)

项目使用 `Verification_code/` 中的验证码图片。原始清单 `train.txt` 和 `val.txt` 每行格式为：

```text
图片文件名 验证码文本
```

验证码文本作为文件名/清单标签，清单中的示例是 5 位。框架会按标签长度把每张图等宽切成字符图，再使用 YOLO 分类模型识别字符，最后按从左到右拼接结果；因此 4 位和 5 位都可以识别。模型学习的是单个字符类别，而不是把每一种验证码组合都当作独立类别。

## 安装

```bash
python -m pip install -r requirements.txt
```

## 准备数据

在项目根目录执行：

```bash
python prepare_dataset.py --source Verification_code --output data/characters --clear
```

输出目录符合 Ultralytics 分类数据集格式：`data/characters/train/<字符>/` 和 `data/characters/val/<字符>/`。原始图片不会被修改。该步骤会生成约 5 倍于验证码图片数量的裁剪图，需要额外磁盘空间。

如果目录中没有清单，或希望直接利用以验证码文本命名的全部图片（包括 4 位验证码），使用：

```bash
python prepare_dataset.py --source Verification_code --output data/characters --discover --clear
```

## 训练

```bash
python train.py --data data/characters --epochs 100 --imgsz 96 --device 0
```

没有 GPU 时使用 `--device cpu`。训练结果默认写入 `runs/captcha/character_classifier/`。

## 预测

```bash
python predict.py Verification_code/ZZVKR.png --positions 5
# 四位验证码：
python predict.py Verification_code/000a.png --positions 4
```

输出验证码和字符平均置信度。预测脚本假设字符等宽，字符数通过 `--positions` 指定（4 位就填 4）。

## 示例图片

`examples/` 中提供 10 张典型验证码图片，用于快速测试数据格式和预测入口。图片文件名本身就是对应标签：

```text
KYHXV.png  6YNE2.png  Y3FV8.png  CHMHX.png  CSZQX.png
5ZXN3.png  C38HM.png  FWM48.png  ZNYXU.png  ZRZQ6.png
```

## 注意事项

- 当前数据集已提供训练/验证划分，准备脚本保持该划分，不随机泄漏样本。
- 如果验证码字符有明显旋转、重叠或非等宽布局，应改为 YOLO 检测方案并为每个字符制作边界框；本数据集的固定尺寸/等宽结构更适合当前分类方案。
- 训练前建议确认 PyTorch 已安装了与显卡 CUDA 匹配的版本。
- 不要对字符分类使用左右翻转增强；例如 `b` 翻转后不再是原字符。训练脚本已关闭此增强，并将默认输入尺寸提高到 96。
