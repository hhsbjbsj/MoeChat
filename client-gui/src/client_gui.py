import flet as ft
import ui
from threading import Thread
import utils

utils_thread = Thread(target=utils.main, args=())
utils_thread.daemon = True
utils_thread.start()

def get_msg_box(msg: str):
    return ft.Container(
        content=ft.Text(msg, size=20, text_align=ft.TextAlign.CENTER),
        padding=5,
        border_radius=10,
        bgcolor=ft.colors.BLUE_900,
    )

def main(page: ft.Page):
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
    page.title = "Moe Chat GUI"

    def send_message_click(e):
        if new_message.value != "" and utils.status:
            mmsg = f"\"{new_message.value}\""
            utils.add_msg_me(mmsg.replace("\"", ""))
            utils.to_llm_and_tts(mmsg, "0.000")
        new_message.value = ""
        new_message.focus()
        page.update()

    # A new message entry form
    new_message = ft.TextField(
        hint_text="输入信息",
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
            content=ui.chat_list,
            border=ft.border.all(1, ft.Colors.OUTLINE),
            border_radius=5,
            padding=10,
            expand=True,
        ),
        ft.Row(
            [
                new_message,
                ft.IconButton(
                    icon=ft.Icons.SEND_ROUNDED,
                    tooltip="Send message",
                    on_click=send_message_click,
                ),
            ]
        ),
    )

ft.app(target=main)