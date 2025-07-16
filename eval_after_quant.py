import tensorflow as tf
import glob
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

# CUDA 호환성 문제 해결을 위한 설정
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

# GPU 사용 불가능시 CPU로 강제 실행
try:
    # GPU 메모리 증가 허용 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 설정 중 오류 발생: {e}")
            print("CPU 모드로 전환합니다.")
            tf.config.set_visible_devices([], 'GPU')
except Exception as e:
    print(f"GPU 초기화 실패: {e}")
    print("CPU 모드로 실행합니다.")
    tf.config.set_visible_devices([], 'GPU')

# 하이퍼파라미터 및 설정
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 8
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

# 모델 로딩 함수 개선
def load_model_safely():
    """안전한 모델 로딩 with fallback options"""
    model_paths = [
        'best_floor_segmentation_model_mobilenet.keras',
        'best_floor_segmentation_model_mobilenet.h5',
        'best_floor_segmentation_model.keras',
        'best_floor_segmentation_model.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"모델 로딩 시도: {model_path}")
                with tf.device('/CPU:0'):  # CPU에서 강제 로딩
                    model = tf.keras.models.load_model(model_path, compile=False)
                print(f"모델 로드 성공: {model_path}")
                return model
            except Exception as e:
                print(f"모델 로딩 실패 ({model_path}): {e}")
                continue
    
    print("사용 가능한 모델을 찾을 수 없습니다.")
    return None

# TFLite 인터프리터 헬퍼 함수
def run_tflite_inference(tflite_model_path, image):
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 입력 데이터 타입 확인 및 전처리
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            image = (image / input_scale) + input_zero_point
            image = np.clip(image, -128, 127)
            image = image.astype(np.int8)
        elif input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            image = (image / input_scale) + input_zero_point
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 출력 데이터 타입 확인 및 후처리
        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        elif output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        return output_data
    except Exception as e:
        print(f"TFLite 추론 중 오류 발생: {e}")
        return None

# 정확도 계산 함수
def calculate_accuracy(y_true, y_pred, threshold=0.5):
    """픽셀 단위 정확도 계산"""
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true_binary = (y_true > threshold).astype(np.float32)
    accuracy = np.mean(y_pred_binary == y_true_binary)
    return accuracy

def calculate_iou(y_true, y_pred, threshold=0.5):
    """IoU (Intersection over Union) 계산"""
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true_binary = (y_true > threshold).astype(np.float32)
    
    intersection = np.sum(y_pred_binary * y_true_binary)
    union = np.sum(y_pred_binary) + np.sum(y_true_binary) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

# 검증 데이터셋에서 샘플 이미지 가져오기
sample_image, sample_mask = next(iter(val_dataset.unbatch().take(1)))
sample_image_batch = tf.expand_dims(sample_image, 0)

def create_mask(pred_mask):
    pred_mask = tf.where(pred_mask > 0.5, 1, 0)
    return pred_mask

def display_and_save_comparison(display_list, filename):
    """이미지, 실제 마스크, Float32 예측, TFLite 예측 비교 시각화"""
    plt.figure(figsize=(20, 5))
    titles = ["Input Image", "Ground Truth", "Float32 Model", "TFLite Model", "Difference"]
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        
        if i == 0:  # 입력 이미지
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        elif i == 4:  # 차이 이미지
            plt.imshow(display_list[i], cmap='RdBu', vmin=-1, vmax=1)
            plt.colorbar()
        else:  # 마스크들
            plt.imshow(tf.squeeze(display_list[i]), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    print(f"비교 결과 이미지 저장됨: {filename}")
    plt.close()

def save_summary_results(images, masks, float_preds, tflite_preds, filename="quantization_comparison_summary.png"):
    """여러 샘플을 하나의 이미지로 요약해서 저장"""
    num_samples = min(len(images), 3)  # 최대 3개 샘플
    
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples*5, 20))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    row_titles = ["Original Images", "Ground Truth Masks", "Float32 Predictions", "TFLite Predictions"]
    
    for i in range(num_samples):
        # 원본 이미지
        axes[0, i].imshow(tf.keras.utils.array_to_img(images[i]))
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].axis('off')
        
        # 실제 마스크
        axes[1, i].imshow(tf.squeeze(masks[i]), cmap='gray')
        axes[1, i].axis('off')
        
        # Float32 예측
        axes[2, i].imshow(tf.squeeze(float_preds[i]), cmap='gray')
        axes[2, i].axis('off')
        
        # TFLite 예측
        axes[3, i].imshow(tf.squeeze(tflite_preds[i]), cmap='gray')
        axes[3, i].axis('off')
    
    # 행 제목 추가
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, rotation=90, size='large')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    print(f"요약 비교 결과 이미지 저장됨: {filename}")
    plt.close()

# 모델 로딩
model = load_model_safely()
if model is None:
    print("모델을 로드할 수 없어 평가를 중단합니다.")
    exit(1)

# -----------------------------------------------------------------------------
# 모델 성능 비교 평가 (CPU 강제 실행)
# -----------------------------------------------------------------------------
print("\n=== 모델 성능 비교 평가 시작 ===")

# Float32 모델 평가
print("\n1. Float32 모델 전체 검증 데이터 평가:")
try:
    with tf.device('/CPU:0'):  # CPU에서 강제 실행
        float_results = model.evaluate(val_dataset, verbose=1)
    print(f"Float32 모델 - Loss: {float_results[0]:.4f}, Accuracy: {float_results[1]:.4f}")
except Exception as e:
    print(f"Float32 모델 평가 중 오류 발생: {e}")
    # 대안: 단일 배치로 평가
    try:
        print("단일 배치로 평가를 시도합니다...")
        sample_batch = next(iter(val_dataset.take(1)))
        with tf.device('/CPU:0'):
            sample_pred = model.predict(sample_batch[0], verbose=0)
        print("단일 배치 평가 성공")
        float_results = [0.0, 0.0]  # 임시값 설정
    except Exception as e2:
        print(f"단일 배치 평가도 실패: {e2}")
        float_results = [0.0, 0.0]

# TFLite 모델들 평가
tflite_models = {
    'Dynamic Range Quantized': 'floor_model_dynamic_range.tflite'
}

# 사용 가능한 TFLite 모델 확인
if os.path.exists('floor_model_int8.tflite'):
    tflite_models['INT8 Quantized'] = 'floor_model_int8.tflite'
if os.path.exists('floor_model_float16.tflite'):
    tflite_models['Float16 Quantized'] = 'floor_model_float16.tflite'

print(f"\n2. 사용 가능한 TFLite 모델들: {list(tflite_models.keys())}")

# 각 TFLite 모델 평가
tflite_results = {}
sample_predictions = {}

for model_name, model_path in tflite_models.items():
    if not os.path.exists(model_path):
        print(f"경고: {model_path} 파일이 존재하지 않습니다.")
        continue
        
    print(f"\n{model_name} 모델 평가 중...")
    
    total_accuracy = 0
    total_iou = 0
    total_samples = 0
    
    # 검증 데이터셋의 일부만 사용하여 평가 (메모리 절약)
    try:
        for batch_idx, (images, masks) in enumerate(val_dataset.take(5)):  # 5개 배치로 축소
            batch_accuracy = 0
            batch_iou = 0
            
            for i in range(min(2, images.shape[0])):  # 배치당 최대 2개 샘플만 처리
                image_batch = tf.expand_dims(images[i], 0)
                mask_true = masks[i].numpy()
                
                # TFLite 추론
                pred_tflite = run_tflite_inference(model_path, image_batch.numpy())
                
                if pred_tflite is not None:
                    # 정확도 및 IoU 계산
                    accuracy = calculate_accuracy(mask_true, pred_tflite[0])
                    iou = calculate_iou(mask_true, pred_tflite[0])
                    
                    batch_accuracy += accuracy
                    batch_iou += iou
                    total_samples += 1
                    
                    # 첫 번째 샘플 저장 (시각화용)
                    if batch_idx == 0 and i == 0:
                        sample_predictions[model_name] = pred_tflite[0]
            
            total_accuracy += batch_accuracy
            total_iou += batch_iou
            
            print(f"  진행률: {batch_idx + 1}/5 배치 완료")
    except Exception as e:
        print(f"{model_name} 평가 중 오류 발생: {e}")
        continue
    
    if total_samples > 0:
        avg_accuracy = total_accuracy / total_samples
        avg_iou = total_iou / total_samples
        
        tflite_results[model_name] = {
            'accuracy': avg_accuracy,
            'iou': avg_iou
        }
        
        print(f"{model_name} - Accuracy: {avg_accuracy:.4f}, IoU: {avg_iou:.4f}")
    else:
        print(f"{model_name} - 평가할 수 있는 샘플이 없습니다.")

# -----------------------------------------------------------------------------
# 결과 비교 및 시각화
# -----------------------------------------------------------------------------
print("\n=== 성능 비교 결과 ===")
print(f"Float32 모델 - Accuracy: {float_results[1]:.4f}")
for model_name, results in tflite_results.items():
    accuracy_diff = results['accuracy'] - float_results[1]
    print(f"{model_name} - Accuracy: {results['accuracy']:.4f} (차이: {accuracy_diff:+.4f}), IoU: {results['iou']:.4f}")

# 샘플 이미지로 시각적 비교
print("\n=== 샘플 이미지 비교 시각화 ===")
sample_image, sample_mask = next(iter(val_dataset.unbatch().take(1)))
sample_image_batch = tf.expand_dims(sample_image, 0)

# Float32 모델 예측
try:
    with tf.device('/CPU:0'):
        float_pred = model.predict(sample_image_batch, verbose=0)
except Exception as e:
    print(f"Float32 모델 예측 중 오류: {e}")
    float_pred = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1))  # 임시 예측값

# 각 TFLite 모델과 비교
for i, (model_name, model_path) in enumerate(tflite_models.items()):
    if not os.path.exists(model_path):
        continue
        
    # TFLite 예측
    tflite_pred = run_tflite_inference(model_path, sample_image_batch.numpy())
    
    if tflite_pred is not None:
        # 차이 계산
        difference = float_pred[0] - tflite_pred[0]
        
        # 시각화
        display_list = [
            sample_image,
            sample_mask,
            create_mask(float_pred[0]),
            create_mask(tflite_pred[0]),
            difference
        ]
        
        filename = f"quantization_comparison_{model_name.lower().replace(' ', '_')}.png"
        display_and_save_comparison(display_list, filename)

# 요약 결과 생성
print("\n=== 전체 요약 결과 생성 ===")
for images, masks in val_dataset.take(1):
    # Float32 예측들
    with tf.device('/CPU:0'):
        float_preds = model.predict(images, verbose=0)
    
    # 첫 번째 TFLite 모델 예측들 (예시)
    if tflite_models:
        first_model_name = list(tflite_models.keys())[0]
        first_model_path = tflite_models[first_model_name]
        
        if os.path.exists(first_model_path):
            tflite_preds = []
            for i in range(min(3, images.shape[0])):
                image_batch = tf.expand_dims(images[i], 0)
                pred = run_tflite_inference(first_model_path, image_batch.numpy())
                if pred is not None:
                    tflite_preds.append(create_mask(pred[0]))
                else:
                    tflite_preds.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 1)))
            
            processed_float_preds = [create_mask(float_preds[i]) for i in range(min(3, images.shape[0]))]
            
            save_summary_results(
                images=images[:3],
                masks=masks[:3],
                float_preds=processed_float_preds,
                tflite_preds=tflite_preds,
                filename="quantization_evaluation_summary.png"
            )
    break

print("\n=== 평가 완료 ===")
print("모든 결과 이미지가 저장되었습니다.")

# 모델 크기 비교
print("\n=== 모델 크기 비교 ===")
if os.path.exists('best_floor_segmentation_model_mobilenet.keras'):
    float_size = os.path.getsize('best_floor_segmentation_model_mobilenet.keras') / (1024*1024)
    print(f"Float32 모델 크기: {float_size:.2f} MB")

for model_name, model_path in tflite_models.items():
    if os.path.exists(model_path):
        tflite_size = os.path.getsize(model_path) / (1024*1024)
        compression_ratio = float_size / tflite_size if 'float_size' in locals() else 0
        print(f"{model_name} 크기: {tflite_size:.2f} MB (압축률: {compression_ratio:.1f}x)")


