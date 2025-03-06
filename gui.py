# -*- coding: utf-8 -*-
# @Time : 2023/5/9 19:49
# @Author : fhp
# @File : gui.py
# @Software : PyCharm
import tkinter as tk
from tkinter import ttk
import test
import train
from transformers import BertTokenizer
from PIL import Image, ImageTk


class Application(tk.Frame):
    """GUI程序的类"""

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)  # 布局，排布显示
        self.createWidget()
        self.tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-uncased/')

    def createWidget(self):
        image = Image.open("background.png")
        resized_image = image.resize((600, 500))
        self.background_image = ImageTk.PhotoImage(resized_image)
        self.theLabel = tk.Label(self.master, justify=tk.LEFT, image=self.background_image,
                                 compound=tk.CENTER, )
        self.theLabel.pack()

        self.wellcome = ttk.Label(self.theLabel, font=('TkDefaultFont', 14))
        self.wellcome.place(x=40, y=30, )
        self.label01 = ttk.Label(self.theLabel, text='Sentence:')
        self.label01.place(x=40, y=100)
        self.label02 = ttk.Label(self.theLabel, text='Aspects:')
        self.label02.place(x=40, y=180)

        self.text01 = tk.Text(self.theLabel, height=5, width=50)
        self.text01.place(x=106, y=100)
        self.text02 = tk.Text(self.theLabel, height=2, width=50)
        self.text02.place(x=106, y=180)

        ttk.Button(self.theLabel, text='开始分析', command=self.process, width=15).place(x=121, y=220)
        ttk.Button(self.theLabel, text='退出', command=self.quit, width=15).place(x=315, y=220)

        ttk.Label(self.theLabel, text='注:Aspects请以\',\'分隔开。', foreground='grey').place(x=106, y=80)

    def process(self):
        x, y = 106, 270
        text = self.text01.get('1.0', 'end-1c')
        words = str(self.text02.get('1.0', 'end-1c'))
        words = words.split(",")
        words = [s.strip() for s in words]
        print(words)
        test.convert(text, words, self.tokenizer)
        polarity = train.Test(self.tokenizer)
        color = []
        color_dict = {'positive': 'green', 'negative': 'red', 'neutral': 'grey'}
        for pol in polarity:
            color.append(color_dict[pol])
        ttk.Label(self.master, text='情感分析结果如下：', font=('宋体', 15, 'bold')).place(x=x, y=y)
        y = y + 40
        for i in range(len(words)):
            ttk.Label(self.master, text=f'{words[i]}: {polarity[i]}', foreground=color[i],
                      font=('Times New Roman', 14)).place(x=x, y=y + i * 40)

# if __name__ == '__main__':
#     tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-uncased')
#
#     root = tk.Tk()
#
#     # 获取屏幕大小
#     screen_width = root.winfo_screenwidth()
#     screen_height = root.winfo_screenheight()
#
#     # 设置窗口大小和位置
#     window_width = 520
#     window_height = 400
#     x = (screen_width - window_width) // 2
#     y = (screen_height - window_height) // 2
#     root.geometry('{}x{}+{}+{}'.format(window_width, window_height, x, y))
#     root.title('方面级情感分析系统')
#
#     app = Application(root)
#
#     root.mainloop()
