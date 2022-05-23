# 人脸关键点检测 + 美颜 + 添加饰品

## 环境
仅限Windows。
```
pip install -r requirements.txt
```

## 用法（生成自argparse）
TLDR：使用-i处理图片，-v处理视频。默认启用美颜。使用-k关闭美颜；使用-m生成人脸网格；使用-c生成对比图；使用-a添加饰品（内置三种，可在`works.py`的`SUPPORTED_MODELS`中自行添加）；使用-o指定输出路径。
```
> python main.py -h
usage: main.py [-h] [-i] [-v] [-k] [-m] [-c] [-r] [-a {glasses,crown,chopper}] [-o OUTPUT] src

Beautify image or video. Large images may take a long time.

positional arguments:
  src                   source file path

options:
  -h, --help            show this help message and exit
  -i, --image           process an image
  -v, --video           process a video
  -k, --keep            keep the face raw. No beautification
  -m, --mesh            generate face mesh
  -c, --compare         concatenate the original image and the processed one, for comparison
  -r, --realtime        when processing video, show realtime result in a window
  -a {glasses,crown,chopper}, --accessory {glasses,crown,chopper}
                        generate an accessory. Supported: dict_keys(['glasses', 'crown', 'chopper'])
  -o OUTPUT, --output OUTPUT
                        specify output file name. For image use .png or .jpg. For video use .mp4 only.
```
