import tensorflow as tf
import glob
import keras_cv
import keras
import matplotlib.pyplot as plt
import numpy as np

# GPU 설정 및 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가 허용 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"사용 가능한 GPU: {len(gpus)}개")
        print(f"GPU 디바이스: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("GPU를 찾을 수 없습니다. CPU로 학습합니다.")

# Mixed Precision 활성화 (GPU 성능 향상)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy.name}")

# 하이퍼파라미터 및 설정
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 8 # GPU 메모리에 따라 조절

# 파일 경로 로드
train_image_paths = sorted(glob.glob('corridor_floor_data/train/images/*.png'))
train_mask_paths = sorted(glob.glob('corridor_floor_data/train/masks/*.png'))
val_image_paths = sorted(glob.glob('corridor_floor_data/test/images/*.png'))
val_mask_paths = sorted(glob.glob('corridor_floor_data/test/masks/*.png'))

print(f"훈련 데이터 수: {len(train_image_paths)}")
print(f"검증 데이터 수: {len(val_image_paths)}")

def load_and_preprocess_image_and_mask(image_path, mask_path):
    """이미지와 마스크를 로드하고 전처리하는 함수"""
    # 이미지 로드 및 정규화
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,(IMG_HEIGHT, IMG_WIDTH))

    # 마스크 로드
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_HEIGHT, IMG_WIDTH), method='nearest')

    # 마스크 이진화 (Binarization)
    # 실제 데이터셋의 마스크는 압축 등으로 인해 0과 255 사이의 회색 값을 가질 수 있습니다.
    # 이를 명확한 0(배경)과 1(바닥)로 변환하는 것은 모델 학습에 매우 중요합니다. [8, 10]
    mask = tf.where(mask > 128, 1, 0)

    # 데이터 타입 변환
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.int32)

    return image, mask

# tf.data.Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

# 데이터 전처리 및 파이프라인 최적화
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.map(load_and_preprocess_image_and_mask, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(load_and_preprocess_image_and_mask, num_parallel_calls=AUTOTUNE)

# 데이터 증강 (선택적이지만 권장)
def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)

# 배치 및 프리페치
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print("tf.data 파이프라인 생성 완료:")
print(train_dataset)

# PASCAL VOC 데이터셋으로 사전 학습된 DeepLabV3+ 모델 로드
# 백본으로 MobileNetV3 Large 사용
# 모델 생성
print("DeepLabV3Plus 사용 가능한 presets:")
print(keras_cv.models.DeepLabV3Plus.presets.keys())

# 사전 훈련된 모델 로드 후 출력층 수정
base_model = keras_cv.models.DeepLabV3Plus.from_preset(
    "deeplab_v3_plus_resnet50_pascalvoc"
)

# 마지막 레이어를 제거하고 새로운 출력층 추가
x = base_model.layers[-2].output  # 마지막 conv layer 전까지

# 512x512로 업샘플링 추가
x = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)  # 128->512
outputs = keras.layers.Conv2D(1, 1, activation='sigmoid', name='binary_output')(x)

model = keras.Model(inputs=base_model.input, outputs=outputs)

# 학습률 스케줄러 설정
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=len(train_image_paths) // BATCH_SIZE * 20 # 예시: 20 에포크
)

# 모델 컴파일 - 바이너리 분할을 위해 손실 함수 변경
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',  # 바이너리 분할용 손실 함수
    metrics=['accuracy', 'binary_accuracy']
)

model.summary()

# 콜백 설정
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_floor_segmentation_model.keras',
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

# GPU 사용 확인 및 모델 학습
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    print(f"학습 디바이스: {'GPU:0' if gpus else 'CPU:0'}")
    
    EPOCHS = 6
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

# 최상의 모델 로드 (콜백으로 저장된 모델)
try:
    best_model = tf.keras.models.load_model('best_floor_segmentation_model.keras')
except:
    # 콜백으로 저장된 모델이 없는 경우 현재 모델 사용
    best_model = model
    best_model.save('best_floor_segmentation_model.keras')

# 검증 데이터셋으로 평가
results = best_model.evaluate(val_dataset)
print("최종 검증 결과:")
for name, value in zip(best_model.metrics_names, results):
    print(f"{name}: {value:.4f}")

def create_mask(pred_mask):
    """로짓(logit) 출력으로부터 이진 마스크 생성"""
    pred_mask = tf.math.sigmoid(pred_mask)
    pred_mask = tf.where(pred_mask > 0.5, 1, 0)
    return pred_mask

def display(display_list):
    """이미지, 실제 마스크, 예측 마스크 시각화"""
    plt.figure(figsize=(15, 5))
    title = ["Input Image", "Ground Truth", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # 마스크는 채널 차원을 제거하여 2D로 표시
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# 검증 데이터셋에서 샘플 배치 가져오기
for image, mask in val_dataset.take(1):
    pred_mask = best_model.predict(image)
    for i in range(min(BATCH_SIZE, 5)): # 최대 5개 샘플 시각화
        display([image[i], mask[i], create_mask(pred_mask[i])])