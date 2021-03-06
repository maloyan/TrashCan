# Оптимизация работы коммунальных служб 🗑️

![](https://us.glasdon.com/images/products/400/glasdon-jubilee-80g-trash-can-3543-silver.jpg)

## Настройка окружения
```
pip install -r requirements.txt
pip install -e .
```
## Запуск обучения
```
python trash/train.py
```

## Конвертация в onnx
```
python trash/torch2onnx.py
```

## Запуск фронта
```
python trash/webapp.py
```

## Запуск фронта через докер
```
docker-compose up --build
```
Зайти на [localhost:8989](http://localhost:8989)

![](images/demo.png)

Завершить работу
```
docker-compose down
```

## Структура проекта
```
├── checkpoints
│   ├── 256_resnet50.onnx
│   └── 256_resnet50.pt
├── data
│   ├── sample_solution.csv
│   ├── test
│   ├── train
│   ├── train.csv
│   ├── Условие_задачи_Чемпионат_Республика_Башкортостан.pdf
│   └── Уфа_baseline.ipynb
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── trash
    ├── configs
    ├── dataset.py
    ├── engine.py
    ├── predict.py
    ├── torch2onnx.py
    ├── train.py
    └── webapp.py
```

## С чем можно поэкспериментировать

- [ ] В конфиге поменять модель на одну из [моделей в бибилиотеке timm](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv)
- [ ] Размеры изображения 256, 384, 512
- [ ] Оптимайзер, шедьюлер, лернинг рейт
