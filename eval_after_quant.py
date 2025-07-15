# TFLite 인터프리터 헬퍼 함수
def run_tflite_inference(tflite_model_path, image):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 입력 데이터 타입이 INT8인 경우 스케일링 필요
    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        image = (image / input_scale) + input_zero_point
        image = np.int8(image)
    
    interpreter.set_tensor(input_details['index'], image)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details['index'])
    
    # 출력 데이터 타입이 INT8인 경우 역-스케일링 필요
    if output_details['dtype'] == np.int8:
        output_scale, output_zero_point = output_details['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    return output_data

# 검증 데이터셋에서 샘플 이미지 가져오기
sample_image, sample_mask = next(iter(val_dataset.unbatch().take(1)))
sample_image_batch = tf.expand_dims(sample_image, 0)

# 각 모델로 추론 실행
float_pred = model.predict(sample_image_batch)
int8_pred = run_tflite_inference('floor_model_int8.tflite', sample_image_batch)

# 결과 시각화
display([
    sample_image, 
    sample_mask, 
    create_mask(float_pred), 
    create_mask(int8_pred)
])
# 시각화 함수 제목 수정 필요
# title =