import telebot
from emotion import *
import cv2
import matplotlib.pyplot as plt

categories = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']

bot = telebot.TeleBot('5489139827:AAEvmUEnWX6J2HKaIMolvS6JskuYrsMLdyk')
print("initialized")


@bot.message_handler(content_types=['photo'])
def get_image_messages(message):
    raw = message.photo[-1].file_id
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('image.png', 'wb') as f:
        f.write(downloaded_file)
    img = cv2.imread('image.png')
    pred = get_emotion(img)
    values = pred.tolist()
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=categories)
    plt.savefig('diagram.png')
    with open('diagram.png', 'rb') as f:
        bot.send_photo(message.chat.id, f.read())


bot.polling(none_stop=True, interval=0)
