# -*- coding: utf-8 -*-
# @Time : 2023/5/10 17:00
# @Author : fhp
# @File : login.py
# @Software : PyCharm
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import gui


class LoginWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("登录")
        # 获取屏幕大小
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.x = (self.screen_width - 600) // 2
        self.y = (self.screen_height - 500) // 2
        self.window.geometry('{}x{}+{}+{}'.format(600, 500, self.x, self.y))

        # 打开图片并调整大小以适应窗口
        image = Image.open("back.png")
        self.background_image = ImageTk.PhotoImage(image)
        theLabel = tk.Label(self.window, justify=tk.LEFT, image=self.background_image,
                            compound=tk.CENTER, )
        theLabel.pack()

        self.username_entry = tk.Entry(self.window)
        self.password_entry = tk.Entry(self.window, show="*")
        self.username_entry.place(x=230, y=245)
        self.password_entry.place(x=230, y=295)

        self.window.bind('<Return>', self.login)
        ttk.Button(self.window, text="登录", command=self.login).place(x=200, y=350)
        ttk.Button(self.window, text="注册", command=self.register).place(x=320, y=350)

    def login(self, event=None):
        # 在这里编写登录逻辑
        username = self.username_entry.get()
        password = self.password_entry.get()

        if len(username) == 0 or len(password) == 0:
            messagebox.showerror("错误", "用户名或密码不能为空")
            return

        # 从文件中读取用户名和密码
        with open("users.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(username + ":"):
                    saved_password = line.split(":")[1]
                    if saved_password == password:
                        print("登录成功")
                        # 登录成功后打开主界面
                        self.window.destroy()

                        root = tk.Tk()

                        # 获取屏幕大小
                        screen_width = root.winfo_screenwidth()
                        screen_height = root.winfo_screenheight()

                        # 设置窗口大小和位置
                        window_width = 600
                        window_height = 500
                        x = (screen_width - window_width) // 2
                        y = (screen_height - window_height) // 2
                        root.geometry('{}x{}+{}+{}'.format(window_width, window_height, x, y))
                        root.title('方面级情感分析系统')

                        app = gui.Application(root)
                        app.wellcome.config(text=f'Hello,{username}!')
                        app.mainloop()
                        return

        messagebox.showerror('错误', '用户名或者密码错误')

    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if len(username) == 0 or len(password) == 0:
            messagebox.showerror("错误", "用户名或密码不能为空")
        else:
            # 将用户名和密码保存到文件中
            with open("users.txt", "a+") as f:
                f.seek(0)  # 将文件指针移动到文件开头
                flag = 0
                for line in f:
                    line = line.strip()
                    if line.startswith(username + ":"):
                        messagebox.showerror("错误", "用户名已存在")
                        flag = 1
                        break
                    # 如果循环没有被 break，则表示用户名不存在，进行写入操作
                if flag == 0:
                    f.write(f"{username}:{password}\n")
                    messagebox.showinfo("提示", "注册成功")

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    login_window = LoginWindow()
    login_window.run()
