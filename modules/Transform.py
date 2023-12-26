import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transform(object):
    def __init__(
            self,
            init_size=224):
        trans = []

        image_size = init_size
        self.strong_transforms_train = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.75),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.75),  # 随机的对比度
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),  # 高斯模糊
                A.GaussNoise(var_limit=(5.0, 30.0)),  # 加高斯噪声
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),  # 桶形 / 枕形畸变
                A.GridDistortion(num_steps=5, distort_limit=1.),  # 网格畸变
                A.ElasticTransform(alpha=3),  # 弹性变换
            ], p=0.4),

            A.CLAHE(clip_limit=4.0, p=0.7),  # 对输入图像应用限制对比度自适应直方图均衡化
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),

            A.Normalize(),
            ToTensorV2()])
        trans.append(self.strong_transforms_train)

        weak = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.75),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.75),  # 随机的对比度

            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(image_size, image_size),

            A.Normalize(),
            ToTensorV2()])

        trans.extend([weak])
        self.trans = trans
        print("in total we have %d transforms" % (len(self.trans)))

    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(image=x)["image"], self.trans))
        return multi_crops
