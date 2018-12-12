from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
#import main

updater = Updater(token='713955229:AAFMCuUwKa6X3SFNqzbP-9dxgOdQeUe6p50')
dispatcher = updater.dispatcher
# Обработка команд
def startCommand(bot, update):
    print("ddddd")
    bot.send_message(chat_id=update.message.chat_id, text='Привет, давай пообщаемся?')
def textMessage(bot, update):
    response = 'Получил Ваше сообщение: ' + update.message.text
    print("ddddd")
    bot.send_message(chat_id=update.message.chat_id, text=response)
print("Хендлеры")
start_command_handler = CommandHandler('start', startCommand)
text_message_handler = MessageHandler(Filters.text, textMessage)
print("Добавляем хендлеры в диспетчер")
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(text_message_handler)
print("Начинаем поиск обновлений")
updater.start_polling(clean=True)
print("Останавливаем бота, если были нажаты Ctrl + C")
updater.idle()
