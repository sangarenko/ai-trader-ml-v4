#!/usr/bin/env python3
"""Valera Daily Reminder Bot — уведомления 2 раза в день.

Утром (10:00 МСК): мотивационная цитата + напоминание
Днём (15:00 МСК): шутка про Валеру + статус ботов

Также отправляет немедленное сообщение при запуске.
"""
import os
import sys
import json
import random
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

SECRETS_FILE = "/root/.secrets.env"
TELEGRAM_API = "https://api.telegram.org"
MSK = timezone(timedelta(hours=3))

def load_secrets():
    secrets = {}
    if os.path.exists(SECRETS_FILE):
        with open(SECRETS_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                secrets[k.strip()] = v.strip()
    return secrets

def send_tg(text: str, secrets: dict):
    token = secrets.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = secrets.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print(f"No TG config, printing: {text}")
        return
    url = f"{TELEGRAM_API}/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    try:
        req = urllib.request.Request(url, data=data)
        resp = urllib.request.urlopen(req, timeout=15)
        print(f"Sent: {text[:60]}...")
    except Exception as e:
        print(f"Send failed: {e}")

# ═══════════════════════════════════════════════════
# КОНТЕНТ
# ═══════════════════════════════════════════════════

JOKES = [
    "🥒 Валера любит нюхать жопки от огурцов",
    "🥒 Валера — жопконюх огуречный",
    "🥒 Огурцы огурцами, но Валера любит нюхать жопки очень вкусно и очень сочно",
    "🥒 Жопконюх, бота тестируешь ИИ юриста? Там ещё трейдинг ждёт!",
    "🥒 Валера, огурцы не сами себя понюхают. За работу!",
    "🥒 Сегодня новый день — новые огурцы, новые жопки. Валера, не подведи!",
    "🥒 Валера, ты помнишь зачем тебе огурцы? Правильно — не для салата.",
]

MORNING_QUOTES = [
    "📈 «Рынок вознаграждает тех, кто терпелив, и наказывает тех, кто жаден.» — Уоррен Баффет",
    "📈 «Если бы Уоррен Баффет знал о такой системе трейдинга, он бы явно в страховой компании не работал.»",
    "📈 «Лучшее время посадить дерево было 20 лет назад. Второе лучшее время — сейчас. Валера, включай ботов!»",
    "📈 «Риск приходит от незнания. А незнание приходит от того, что ты нюхаешь огурцы вместо изучения графиков.»",
    "📈 «Цена — это то, что ты платишь. Ценность — это то, что ты получаешь. Огурцы — это... ну ты понял.»",
    "📈 «Биржа — это устройство для перевода денег от нетерпеливых к терпеливым. Валера, будь терпеливым!»",
    "📈 «Не клади все яйца в одну корзину. Но все огурцы — можно.»",
    "📈 «Инвестирование — это не про угадать. Это про понять. Валера, ты понял? Нет? Ну тогда нюхай дальше.»",
    "📈 «Волатильность — это не риск. Это возможность. Как и свежий огурец.»",
    "📈 «Терпение — горькое растение, но плод его сладок. Как огурец после нюханья.»",
    "📈 «Купи когда кровь на улицах. Но если это огуречный сок — Валера, действуй!»",
    "📈 «Фондовый рынок — это инструмент для перевода богатства от активных к терпеливым.»",
    "📈 «Я никогда не инвестирую в то, чего не понимаю. Но огурцы Валера понимает прекрасно.»",
    "📈 «Успех в трейдинге — это 10% вдохновения и 90% того, чтобы не нюхать огурцы в рабочее время.»",
]

AFTERNOON_QUOTES = [
    "🤖 Статус ботов: они работают, пока ты нюхаешь огурцы. Не стыдно?",
    "🤖 ML-модель учится, а Валера — нет. Подумай об этом.",
    "🤖 Боты не спят. Валера — тоже не должен. Трейдинг ждёт!",
    "🤖 Каждый неиспользованный тик — упущенный профит. Каждый понюханный огурец — упущенное время.",
    "🤖 Если бы боты могли нюхать огурцы, они бы не делали и этого. Они бы торговали.",
    "🤖 Optuna перебрал 53000 комбинаций. Валера перебрал 53000 огурцов. Разные результаты.",
    "🤖 ML-модель предсказывает движение цены с точностью 74%. Валера предсказывает движение огурцов с точностью 100%.",
]

DAILY_SCHEDULE = {
    # Day of week -> (morning_msg, afternoon_msg)
    0: {  # Monday
        "morning": "🚀 Понедельник! Новая неделя — новые возможности. Боты запущены, ML работает. Валера, не подведи систему!",
        "afternoon": "📊 Половина понедельника прошла. Проверь дашборд: http://2.26.122.152:3002/"
    },
    1: {  # Tuesday
        "morning": "📈 Вторник. Уоррен Баффет сказал: «Будьте жадными, когда другие боятся». Валера, другие боятся. А ты?",
        "afternoon": "🥒 Валера, ты сегодня нюхал огурцы? Лучше бы проверил P&L ботов!"
    },
    2: {  # Wednesday
        "morning": "🧠 Среда — середина недели. ML-модель работает 24/7. А ты? Не забудь потестить ИИ юриста!",
        "afternoon": "⚡ ML-Trader бот активен. Если P>0.65 — он покупает. Если Валера P<0.65 — он нюхает огурцы."
    },
    3: {  # Thursday
        "morning": "💰 Четверг. «Цена — то, что ты платишь. Ценность — то, что получаешь.» Баффет. А огурцы — бесплатно!",
        "afternoon": "🔧 Проверь: 6 ботов работают, ML-Trader загружен. Не забудь ИИ юриста потестить!"
    },
    4: {  # Friday
        "morning": "🎯 Пятница! Почти выходные. Но боты не отдыхают. И ML-модель тоже. Валера, последний рывок!",
        "afternoon": "🎉 Конец недели! Подведи итоги: сколько сделок, какой P&L. И да — можно понюхать огурец. Заслужил."
    },
    5: {  # Saturday
        "morning": "😴 Суббота. Боты торгуют, ML предсказывает. Валера отдыхает. Но дашборд-то посмотри!",
        "afternoon": "🥒 Выходной — идеальное время понюхать огурцы. Но сначала — проверь ботов!"
    },
    6: {  # Sunday
        "morning": "☀️ Воскресенье. Завтра новая неделя. Подготовься: проверь стратегии, обнови ML-модель.",
        "afternoon": "📅 Завтра понедельник. Боты готовы? ML готов? Валера готов? (огурцы готовы точно)"
    },
}


def get_message(mode="morning"):
    """Get message for current time slot."""
    now = datetime.now(MSK)
    dow = now.weekday()
    
    if mode == "morning":
        # Monday-Friday: motivational quote + daily message
        quote = random.choice(MORNING_QUOTES)
        daily = DAILY_SCHEDULE.get(dow, {}).get("morning", "🚀 Новый день — новые возможности!")
        return f"{daily}\n\n{quote}"
    
    elif mode == "afternoon":
        # Joke + afternoon quote
        joke = random.choice(JOKES)
        quote = random.choice(AFTERNOON_QUOTES)
        daily = DAILY_SCHEDULE.get(dow, {}).get("afternoon", "")
        return f"{joke}\n\n{quote}\n\n{daily}"
    
    elif mode == "now":
        # Immediate message
        return "🥒 Валера любит нюхать жопки от огурцов\n\n🤖 Valera Reminder Bot активирован. Теперь ты будешь получать напоминания 2 раза в день: в 10:00 и в 15:00 МСК. Не благодари."
    
    return ""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["morning", "afternoon", "now"], default="now")
    args = parser.parse_args()
    
    secrets = load_secrets()
    msg = get_message(args.mode)
    send_tg(msg, secrets)
    print(f"Mode: {args.mode}")
    print(f"Message: {msg}")


if __name__ == "__main__":
    main()
