## Хакатон Digital League AI Challenge

#### Настройка окружения
```
pip install -e .
pip install -r requirements
```
#### Запуск обучения
```
python trash/train.py configs/config.json 
```

#### Конвертация в onnx
```
python trash/torch2onnx.py configs/config.json 
```

#### Запуск фронта
```
python trash/webapp.py checkpoints/128_resnet50.onnx
```

#### Собрать докер
```
docker build . -t trash
docker run -p 8989:8989 --ipc=host -d --rm trash:latest
```