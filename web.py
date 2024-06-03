import flet as ft
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
config = PeftConfig.from_pretrained("SouthMemphis/Saiga-lora-2048-2epochs")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path , cache_dir="/", load_in_4bit=True)
model = PeftModel.from_pretrained(model, "SouthMemphis/Saiga-lora-2048-2epochs", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("SouthMemphis/Saiga-lora-2048-2epochs", cache_dir="/")

global model_message_sent
model_message_sent = False

class Message():
    def __init__(self, user_name: str, text: str, message_type: str):
        self.user_name = user_name
        self.text = text
        self.message_type = message_type

class ChatMessage(ft.Row):
    def __init__(self, message: Message):
        super().__init__()
        message_style = {
            "background-color": "#ffcccc",
            "border-radius": "10px",
            "padding": "10px",
        }
        self.vertical_alignment="start"
        self.controls=[
                ft.CircleAvatar(
                    content=ft.Text(self.get_initials(message.user_name)),
                    color=ft.colors.WHITE,
                    bgcolor=self.get_avatar_color(message.user_name),
                ),
                ft.Column(
                    [
                        ft.Text(message.user_name, weight="bold", color='black'),
                        ft.Text(message.text, selectable=True, width=1900, color='black')


                    ],

                    tight=True,
                    spacing=5,

                ),
            ]

    def get_initials(self, user_name: str):
        if user_name:
            return user_name[:1].capitalize()
        else:
            return "Unknown"  

    def get_avatar_color(self, user_name: str):
        colors_lookup = [
            ft.colors.AMBER,
            ft.colors.BLUE,
            ft.colors.BROWN,
            ft.colors.CYAN,
            ft.colors.GREEN,
            ft.colors.INDIGO,
            ft.colors.LIME,
            ft.colors.ORANGE,
            ft.colors.PINK,
            ft.colors.PURPLE,
            ft.colors.RED,
            ft.colors.TEAL,
            ft.colors.YELLOW,
        ]
        return colors_lookup[hash(user_name) % len(colors_lookup)]

def main(page: ft.Page):
    page.horizontal_alignment = "stretch"
    page.title = "Flet Chat"
    def join_chat_click(e):
        if not join_user_name.value:
            join_user_name.error_text = "Name cannot be blank!"
            join_user_name.update()
        else:
            page.session.set("user_name", join_user_name.value)
            page.dialog.open = False
            new_message.prefix = ft.Text(f"{join_user_name.value}: ")
            page.pubsub.send_all(Message(user_name=join_user_name.value, text=f"{join_user_name.value} has joined the chat.", message_type="login_message"))
            page.update()
        bot_message = "Привет, я - нейросетевая модель Saiga, дообученная пользователем Hugging Face под ником SouthMemphis.\nЯ умею выполнять суммаризацию новостей на русском языке!\nОтправь в чат текст, а я тебе выдам краткую суммаризацию."

        page.pubsub.send_all(Message('Model', bot_message, message_type="chat_message"))




    

    model_message_sent = False

    def send_message_click(e):

        
        if new_message.value != "":
            user_input = 'Тебе на вход поступает русскоязычная статья из газеты. Твоя задача - выполнить суммаризацию этой статьи. Выдели из статьи наиболее релевантные фрагменты и по ним составь её суммаризацию.\n' + 'Статья:\n' + new_message.value + '\nЕё суммаризация:'
            page.pubsub.send_all(Message(page.session.get("user_name"), new_message.value, message_type="chat_message"))
            new_message.value = ""
            new_message.focus()
            page.update()
            model.eval()
            with torch.no_grad():
                result = tokenizer.decode(
                    model.generate(input_ids=tokenizer.encode(user_input, return_tensors="pt"), max_new_tokens=700)[0],
                    skip_special_tokens=False)
                bot_response = result[result.find("Её суммаризация:") + len("Её суммаризация:") + 1:result.find("</s><s>")-3]

            page.pubsub.send_all(Message('Model', bot_response, message_type="chat_message"))
            new_message.value = ""
            new_message.focus()
            page.update()


    def on_message(message: Message):
        if message.message_type == "chat_message":
            m = ChatMessage(message)
        elif message.message_type == "login_message":
            m = ft.Text(message.text, italic=True, color=ft.colors.RED, size=12)
        chat.controls.append(m)
        page.update()

    page.pubsub.subscribe(on_message)

    # A dialog asking for a user display name
    join_user_name = ft.TextField(
        label="Введите своё имя",
        autofocus=True,
        on_submit=join_chat_click,
    )
    page.dialog = ft.AlertDialog(
        open=True,
        modal=True,
        title=ft.Text("Добро пожаловать!"),
        content=ft.Column([join_user_name], width=300, height=70, tight=True),
        actions=[ft.ElevatedButton(text="Join chat", on_click=join_chat_click)],
        actions_alignment="end",
    )

    # Chat messages
    chat = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
    )

    # A new message entry form
    new_message = ft.TextField(
        hint_text="Напишите сообщение...",
        autofocus=True,
        shift_enter=True,
        min_lines=1,
        max_lines=5,
        filled=True,
        expand=True,
        on_submit=send_message_click,

    )

    # Add everything to the page
    page.add(
        ft.Container(
            content=chat,
            border=ft.border.all(1, ft.colors.OUTLINE),
            border_radius=5,
            padding=10,
            expand=True,
            bgcolor='white'
        ),
        ft.Row(
            [
                new_message,
                ft.IconButton(
                    icon=ft.icons.SEND_ROUNDED,
                    tooltip="Send message",
                    on_click=send_message_click,

                ),
            ]
        ),
    )

ft.app(port=8550, target=main, view=ft.WEB_BROWSER)
