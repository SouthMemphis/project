# Проект по автоматической суммаризации текстов с применением нейронных сетей
## Структура репозитория
* fine-tuning.ipynb - файл с  отчётом и описанием процесса тонкой настройки LLM под домен.
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

eval_prompt = f"""Тебе на вход поступает русскоязычная статья из газеты. Твоя задача - выполнить суммаризацию этой статьи. Выдели из статьи наиболее релевантные фрагменты и по ним составь её суммаризацию.

### Статья:
{<Ваша новостная статья>}

### Её суммаризация:"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    result = tokenizer.decode(model.generate(**model_input, max_new_tokens=600)[0], skip_special_tokens=False)
    eval = result[result.find("Её суммаризация:") + len("Её суммаризация:") + 1:result.find("</s><s>") - 3]
    print(eval)
```
Примеры работы модели можно найти в файле fine-tuning.ipynb

