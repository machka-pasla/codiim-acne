import sqlite3
from datetime import datetime

conn = sqlite3.connect('DB.db')

cursor = conn.cursor()
def start ():
    conn = sqlite3.connect('DB.db')
# bug - бюджет пользователя (максимум по цене на лекарство)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                       id INTEGER PRIMARY KEY,
                       name TEXT,
                       tg_nik TEXT,
                       bug INTEGER,
                       time TEXT
                   )''')

# таблица запросов в текст - рекомендации (финальный ответ нейронки, список лекарств (соответствующих ид из бд)) , pain - заболевания , time - время и дата когда был добавлен запрос
    cursor.execute('''CREATE TABLE IF NOT EXISTS ZZZ (
                       time TEXT PRIMARY KEY,
                       us_id INTEGER,
                       text TEXT,
                       pain TEXT
                   )''')

# дбавляет нового юза в таблицу : chat_id - понятно что , name_tg - имя пользоывателя в тг , tg_nik - краткий ник (которы начинается с @)
# bug - бюджет == -1 => не задан (=неогрничен)
def add_us (chat_id,name_tg,tgnik,bug):
    date = datetime.now().date()
    time=datetime.now()
    date = str(date)+"|"+str(time)
    cursor.execute('INSERT INTO users (id, name, tg_nik,bug,time) VALUES (?, ?, ?, ?, ?)', (chat_id, name_tg, tgnik,bug,date))

def add_hape (chat_id,text,pain):
    date = datetime.now().date()
    time=datetime.now()
    date = str(date)+"|"+str(time)
    cursor.execute('INSERT INTO ZZZ (time, us_id, text,pain) VALUES (?, ?, ?, ?)', (date,chat_id, text, pain))

def rebudget (chat_id,bug):
    cursor.execute('UPDATE users SET bug = ? WHERE id = ?', (bug,chat_id))

def get_buget (chat_id):
    cursor.execute('SELECT bug FROM users WHERE id = ?', (chat_id,))
    results = cursor.fetchone()
    results=results[0]
    return results

def get_user (chat_id):
    cursor.execute('SELECT * FROM users WHERE id = ?', (chat_id,))
    results = cursor.fetchone()
    return results

def get_all_user_ZZZ (chat_id):
    cursor.execute('SELECT * FROM ZZZ WHERE us_id = ?', (chat_id,))
    results = cursor.fetchone()
    return results