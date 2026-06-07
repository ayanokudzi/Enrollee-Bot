"""Telegram bot for the FCS HSE admission RAG assistant.

Setup:
  1) Get a token from @BotFather in Telegram.
  2) export TELEGRAM_BOT_TOKEN="your-token"
  3) python build_index.py        # once, builds the knowledge base
  4) python bot.py
"""
import asyncio
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          ContextTypes, filters)

import config
from rag import RAGPipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enrollee-bot")

WELCOME = (
    "Здравствуйте! Я бот-ассистент приёмной комиссии Факультета компьютерных наук "
    "НИУ ВШЭ. Задайте вопрос о поступлении — программах, сроках, документах, льготах "
    "или вступительных испытаниях.\n\n"
    "Пожалуйста, не вводите персональные данные (ФИО, паспорт, СНИЛС): я их не запрашиваю "
    "и не храню. Мои ответы носят справочный характер."
)

pipeline: RAGPipeline = None  # loaded in main()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME)


async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    # generation is CPU/GPU heavy — run it off the event loop
    loop = asyncio.get_event_loop()
    try:
        answer = await loop.run_in_executor(None, pipeline.answer, query)
    except Exception as e:
        log.exception("answer failed")
        answer = ("Произошла ошибка при обработке запроса. Попробуйте переформулировать "
                  f"вопрос или обратитесь в приёмную комиссию: {config.CONTACT}")
    await update.message.reply_text(answer, parse_mode="Markdown")


def main():
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env variable (token from @BotFather).")

    global pipeline
    log.info("Loading RAG pipeline (models + index)...")
    pipeline = RAGPipeline()
    log.info("Pipeline ready.")

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    log.info("Bot started. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
