import tensorflow as tf

# 최상의 Keras 모델 로드
model = tf.keras.models.load_model('best_floor_segmentation_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#############################
## Float32 TFLite 모델로 변환 ##
#############################
'''
tflite_model = converter.convert()

# TFLite 모델 파일로 저장
with open('floor_model_float32.tflite', 'wb') as f:
    f.write(tflite_model)

print("Float32 TFLite 모델 변환 완료.")
'''

##################
## 동적 범위 양자화 ##
##################

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('floor_model_dynamic_range.tflite', 'wb') as f:
    f.write(tflite_model)

print("동적 범위 양자화 TFLite 모델 변환 완료.")