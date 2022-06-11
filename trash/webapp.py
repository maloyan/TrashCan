import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf
from pkg_resources import resource_filename

config = OmegaConf.load(resource_filename(__name__, "configs/config.yaml"))
# Constants

DEVICE = "cpu"
IMG_SIZE = config.image_size
PORT = 8989
TARGET_DICT = {
    0: "неудовлетворительное качество",
    1: "соответствует всем необходимым условиям",
    2: "отсутствует мусорный бак",
}
INSTRUCTION = """### Инструкция
1. Выберите изображение и загрузите его
2. Нажмите на кнопку "Submit" для распознавания. Или же нажмите "Clear" чтобы выбрать другое изображение.
"""
model = ort.InferenceSession(
    f"{config['checkpoints']}/{config['image_size']}_{config['model']}.onnx"
)


def handler(orig_image):
    image = cv2.resize(orig_image, (IMG_SIZE, IMG_SIZE))
    image = np.array([np.moveaxis(image, -1, 0)], dtype=np.float32)

    output = model.run(None, {"image": image})
    output = output[0].argmax()

    return TARGET_DICT[output]

iface = gr.Interface(
    handler,
    inputs=gr.inputs.Image(label="Загруженное изображение", type = 'numpy'),
    outputs=gr.outputs.Textbox(label="Результат"),
    title="Оптимизация работы коммунальных служб 🗑️",
    article=INSTRUCTION,
)
iface.launch(server_port=PORT, server_name="0.0.0.0", enable_queue=True, share=True)
