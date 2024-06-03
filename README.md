# Проект по автоматической суммаризации текстов с применением нейронных сетей
## Структура репозитория
* fine-tuning.ipynb - файл с теоретическим отчётом и описанием процесса тонкой настройки LLM под домен.
* requirments.txt - библиотеки из окружения.
* web.py - Веб-интерфейс приложения, использующего настроенную модель 
* Video_2024_05_15-1.webm - демонстрация работы приложения</br>
Набор данных, на котором производилось обучение и оценка качества: IlyaGusev/gazeta

### Метрики качества обученной модели
![изображение](https://github.com/SouthMemphis/project/assets/92672290/58b5f068-8deb-4f2b-b9d8-2d3b6a8e1fc1)
### Использование модели из Hugging face:
```
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("SouthMemphis/Saiga-lora-2048-2epochs")
base_model = AutoModelForCausalLM.from_pretrained("IlyaGusev/saiga_mistral_7b_merged")
model = PeftModel.from_pretrained(base_model, "SouthMemphis/Saiga-lora-2048-2epochs")
```
## Как воспользоваться проектом с Web-интерфейсом?
