import os
import sqlite3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from sklearn.linear_model import LogisticRegression

# Load environment variables
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# Initialize database
conn = sqlite3.connect("daman_bot.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, balance INTEGER DEFAULT 10000, bet_history TEXT)"""
)
conn.commit()

# Train AI Model
data = pd.DataFrame(
    {
        "prev_1": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        "prev_2": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        "prev_3": [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        "result": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    }
)

X_train = data[["prev_1", "prev_2", "prev_3"]]
y_train = data["result"]
model = LogisticRegression()
model.fit(X_train, y_train)


# Helper Functions
def get_user_data(user_id):
    cursor.execute("SELECT balance, bet_history FROM users WHERE user_id=?", (user_id,))
    user = cursor.fetchone()
    if user:
        return user[0], user[1]
    else:
        cursor.execute("INSERT INTO users (user_id, balance, bet_history) VALUES (?, ?, ?)", (user_id, 10000, ""))
        conn.commit()
        return 10000, ""


def update_user_data(user_id, balance, bet_history):
    cursor.execute("UPDATE users SET balance=?, bet_history=? WHERE user_id=?", (balance, bet_history, user_id))
    conn.commit()


# Telegram Bot Commands
def start(update: Update, context: CallbackContext) -> None:
    user_id = update.message.chat_id
    balance, _ = get_user_data(user_id)
    update.message.reply_text(
        f"🎲 Welcome to Daman Predictor Bot! 🎲\n"
        f"💰 Your balance: ₹{balance}\n"
        f"🔮 Use: `/predict Big Small Big Small | Big`"
    )


def predict(update: Update, context: CallbackContext) -> None:
    user_id = update.message.chat_id
    balance, bet_history = get_user_data(user_id)

    if len(context.args) < 3 or "|" not in context.args:
        update.message.reply_text("⚠ Format error! Use: `/predict Big Small Big Small | Big`")
        return

    args = " ".join(context.args).split("|")
    results = args[0].strip().split()
    last_result = args[1].strip()

    if len(results) < 3:
        update.message.reply_text("⚠ Enter at least 3 past results.")
        return

    numeric_results = [1 if x.lower() == "big" else 0 for x in results[-3:]]
    prediction = model.predict([numeric_results])[0]
    confidence = model.predict_proba([numeric_results])[0][prediction] * 100

    prediction_text = "🔥 Next Bet: Big" if prediction == 1 else "🔥 Next Bet: Small"

    fibonacci_sequence = [100, 100, 200, 300, 500, 800, 1300, 2100]
    bet_suggestion = fibonacci_sequence[min(len(results), len(fibonacci_sequence) - 1)]

    bet_history += f"{last_result} → {prediction_text}\n"
    balance -= bet_suggestion
    if prediction == (1 if last_result.lower() == "big" else 0):
        balance += bet_suggestion * 2

    update_user_data(user_id, balance, bet_history)

    response = (
        f"{prediction_text}\n"
        f"📊 Confidence: {confidence:.2f}%\n"
        f"💰 Bet Amount: ₹{bet_suggestion}\n"
        f"💵 Balance: ₹{balance}\n"
        f"📜 Recent Bets:\n{bet_history[-200:]}"
    )

    update.message.reply_text(response)


def reset(update: Update, context: CallbackContext) -> None:
    user_id = update.message.chat_id
    update_user_data(user_id, 10000, "")
    update.message.reply_text("🔄 Balance reset to ₹10,000!")


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("reset", reset))
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
