import tensorflow as tf
import glob
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

# GPU 설정 및 메모리 제한
try:
    # GPU 메모리 증가 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU 메모리 성장 허용
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 사용 가능: {len(gpus)}개")
        except RuntimeError as e:
            print(f"GPU 설정 중 오류 발생: {e}")
    else:
        print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
except Exception as e:
    print(f"GPU 초기화 오류: {e}")
    print("CPU로 폴백합니다.")

# CUDA 버전 호환성 문제로 인해 CPU 강제 사용
print("CUDA 호환성 문제로 인해 CPU를 강제로 사용합니다.")
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 설정 (재현 가능한 결과를 위해)
tf.random.set_seed(42)
np.random.seed(42)

# 하이퍼파라미터 및 설정
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 4  # CPU 사용으로 인한 배치 크기 감소
NUM_CLASSES = 1

# 파일 경로 로드
train_image_paths = sorted(glob.glob('corridor_floor_data/train/images/*.png'))
train_mask_paths = sorted(glob.glob('corridor_floor_data/train/masks/*.png'))
val_image_paths = sorted(glob.glob('corridor_floor_data/test/images/*.png'))
val_mask_paths = sorted(glob.glob('corridor_floor_data/test/masks/*.png'))

print(f"훈련 데이터 수: {len(train_image_paths)}")
print(f"검증 데이터 수: {len(val_image_paths)}")

# -----------------------------------------------------------------------------
# 데이터 로딩 및 전처리 파이프라인
# -----------------------------------------------------------------------------
def load_and_preprocess_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_HEIGHT, IMG_WIDTH), method='nearest')
    mask = tf.where(mask > 128, 1, 0)

    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.int32)
    return image, mask

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

train_dataset = train_dataset.map(load_and_preprocess_image_and_mask, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(load_and_preprocess_image_and_mask, num_parallel_calls=AUTOTUNE)

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print("tf.data 파이프라인 생성 완료:")
print(train_dataset)

# -----------------------------------------------------------------------------
# DeepLabV3+ 모델 직접 구현 (MobileNetV2 백본 사용)
# -----------------------------------------------------------------------------
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    """표준 Convolution 블록"""
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    # ❗️❗️❗️ 수정된 부분: tf.nn.relu 대신 layers.ReLU() 사용 ❗️❗️❗️
    x = layers.ReLU()(x)
    return x

def ASPP(image_features):
    """Atrous Spatial Pyramid Pooling (ASPP) 블록"""
    shape = image_features.shape
    y_pool = layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = convolution_block(y_pool, kernel_size=1)
    y_pool = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = convolution_block(image_features, kernel_size=1, dilation_rate=1)
    y_6 = convolution_block(image_features, kernel_size=3, dilation_rate=6)
    y_12 = convolution_block(image_features, kernel_size=3, dilation_rate=12)
    y_18 = convolution_block(image_features, kernel_size=3, dilation_rate=18)

    y = layers.Concatenate()([y_pool, y_1, y_6, y_12, y_18])
    y = convolution_block(y, kernel_size=1)
    return y

def DeeplabV3Plus(image_size, num_classes):
    """MobileNetV2 백본을 사용하는 DeepLabV3+ 모델 정의"""
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    backbone = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=model_input
    )

    high_level_features = backbone.get_layer("block_13_expand_relu").output
    x = ASPP(high_level_features)

    low_level_features = backbone.get_layer("block_6_expand_relu").output
    low_level_features = convolution_block(low_level_features, num_filters=48, kernel_size=1)

    x = layers.UpSampling2D(
        size=(low_level_features.shape[1] // x.shape[1], low_level_features.shape[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    x = layers.Concatenate()([x, low_level_features])
    x = convolution_block(x)
    x = convolution_block(x)

    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation="sigmoid")(x)

    return keras.Model(inputs=model_input, outputs=model_output)

# 모델 생성
print("CPU에서 MobileNetV2 백본을 사용한 DeepLabV3+ 모델 생성")
model = DeeplabV3Plus(image_size=IMG_HEIGHT, num_classes=NUM_CLASSES)
model.summary()

# -----------------------------------------------------------------------------
# 모델 컴파일 및 학습
# -----------------------------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_floor_segmentation_model_mobilenet.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

EPOCHS = 20  # CPU 훈련으로 인한 에포크 수 감소 (원래 50)
print(f"\n경고: CPU에서 훈련하므로 속도가 느릴 수 있습니다.")
print(f"에포크 수를 {EPOCHS}로 줄였습니다. GPU 문제 해결 후 50으로 증가시키세요.")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------------------------------------------------------
# 평가 및 시각화
# -----------------------------------------------------------------------------
try:
    best_model = tf.keras.models.load_model('best_floor_segmentation_model_mobilenet.keras')
except Exception as e:
    print(f"저장된 모델을 불러오는 데 실패했습니다: {e}")
    best_model = model

print("\n최종 검증 결과:")
results = best_model.evaluate(val_dataset)
for name, value in zip(best_model.metrics_names, results):
    print(f"{name}: {value:.4f}")

def create_mask(pred_mask):
    pred_mask = tf.where(pred_mask > 0.5, 1, 0)
    return pred_mask

def display(display_list):
    plt.figure(figsize=(15, 5))
    title = ["Input Image", "Ground Truth", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image, mask in val_dataset.take(1):
    pred_mask = best_model.predict(image)
    for i in range(min(BATCH_SIZE, 5)):
        display([image[i], mask[i], create_mask(pred_mask[i])])