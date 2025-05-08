import os
import sqlite3
import hmac
import hashlib
import uuid
from datetime import datetime as dt_obj # Переименовал импорт, чтобы избежать конфликта с модулем datetime
from flask import Flask, render_template, request, redirect, session, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json # Импортируем json здесь

# Предполагаем, что GET_predict теперь возвращает словарь
from back import GET_predict

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
BOT_USERNAME = os.getenv('TELEGRAM_BOT_USERNAME')
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key_for_development_123!@#')

app = Flask(__name__)
app.secret_key = SECRET_KEY

DB_PATH = 'DB.db'
UPLOAD_FOLDER = 'photos'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Регистрация кастомного фильтра Jinja2 ---
def format_datetime_filter(value, format_str='%Y-%m-%d %H:%M:%S'):
    if value is None:
        return "N/A"
    try:
        # Убираем информацию о часовом поясе, если она есть (для fromisoformat)
        # и обрабатываем возможные микросекунды
        if '+' in value:
            value = value.split('+')[0]
        if 'Z' in value: # UTC Zulu time
            value = value.replace('Z', '')

        if '.' in value:
            # Отбрасываем микросекунды, если они есть, так как strptime может их не ожидать по умолчанию
            # или fromisoformat может с ними работать напрямую
            dt = dt_obj.fromisoformat(value.split('.')[0])
        else:
            dt = dt_obj.fromisoformat(value)
        return dt.strftime(format_str)
    except (ValueError, TypeError) as e:
        # print(f"Error formatting date '{value}': {e}") # Для отладки
        return value # Возвращаем оригинальное значение, если парсинг не удался

app.jinja_env.filters['format_datetime'] = format_datetime_filter
# --- Конец регистрации фильтра ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        tg_id INTEGER UNIQUE NOT NULL,
        first_name TEXT,
        username TEXT,
        auth_date TEXT
      )
    ''')
    c.execute('''
      CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        files TEXT, 
        analysis_skin_type TEXT,
        analysis_acne TEXT,
        analysis_rosacea TEXT,
        analysis_comedones TEXT,
        final_recommendation TEXT, 
        upload_time TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
      )
    ''')
    conn.commit()
    conn.close()

def verify_telegram(data: dict) -> bool:
    if 'hash' not in data: return False
    check_hash = data.pop('hash')
    data_check_arr = [f"{k}={v}" for k, v in sorted(data.items())]
    data_check_string = "\n".join(data_check_arr)
    if not BOT_TOKEN:
        print("CRITICAL: TELEGRAM_BOT_TOKEN is not set.")
        return False
    secret_key_bytes = hashlib.sha256(BOT_TOKEN.encode()).digest()
    hmac_hash = hmac.new(secret_key_bytes, data_check_string.encode(), hashlib.sha256).hexdigest()
    return hmac_hash == check_hash

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def get_file_hash(file_storage):
    original_filename = secure_filename(file_storage.filename)
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'dat'
    hasher = hashlib.sha256()
    chunk_size = 8192
    current_pos = file_storage.tell()
    file_storage.seek(0)
    while chunk := file_storage.read(chunk_size):
        hasher.update(chunk)
    file_storage.seek(current_pos)
    unique_suffix = uuid.uuid4().hex[:8] 
    return f"{hasher.hexdigest()}_{unique_suffix}.{ext}"

@app.route('/')
def index():
    return render_template('index.html', bot_username=BOT_USERNAME)

@app.route('/login', methods=['GET'])
def login():
    data = request.args.to_dict()
    if not BOT_TOKEN:
        flash('Ошибка конфигурации сервера Telegram.', 'danger')
        return redirect(url_for('index'))
    if not verify_telegram(data.copy()):
        flash('Не удалось пройти аутентификацию через Telegram.', 'danger')
        return redirect(url_for('index'))
    
    tg_id = int(data['id'])
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO users (tg_id, first_name, username, auth_date) VALUES (?, ?, ?, ?)',
              (tg_id, data.get('first_name'), data.get('username'), data.get('auth_date')))
    conn.commit()
    c.execute('SELECT id FROM users WHERE tg_id = ?', (tg_id,))
    user_db_id_tuple = c.fetchone()
    conn.close()

    if user_db_id_tuple:
        session['user_db_id'] = user_db_id_tuple[0]
        session['tg_id'] = tg_id
        session['first_name'] = data.get('first_name', 'Пользователь')
        return redirect(url_for('dashboard'))
    else:
        flash('Не удалось создать или найти пользователя в базе данных.', 'danger')
        return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_db_id' not in session:
        flash('Пожалуйста, войдите в систему.', 'warning')
        return redirect(url_for('index'))

    user_db_id = session['user_db_id']

    if request.method == 'POST':
        photos_files = [request.files.get(f'photo{i}') for i in (1, 2, 3)]

        if not all(p_file and p_file.filename and allowed_file(p_file.filename) for p_file in photos_files):
            flash('Пожалуйста, загрузите три фото в формате JPG, PNG или JPEG.', 'warning')
            return redirect(request.url)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        saved_file_paths_for_processing = []
        hashed_filenames_for_db = []

        for i, p_file in enumerate(photos_files):
            hashed_filename = get_file_hash(p_file)
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], hashed_filename)
            try:
                p_file.seek(0) 
                p_file.save(full_path)
                saved_file_paths_for_processing.append(full_path)
                hashed_filenames_for_db.append(hashed_filename)
            except Exception as e:
                flash(f'Ошибка при сохранении файла {secure_filename(p_file.filename)}: {str(e)}', 'danger')
                return redirect(request.url)
        
        files_json_for_db = json.dumps(hashed_filenames_for_db)

        try:
            analysis_results = GET_predict(saved_file_paths_for_processing)
        except Exception as e:
            flash(f'Ошибка при анализе изображений: {str(e)}', 'danger')
            return redirect(request.url)

        now_iso = dt_obj.now().isoformat(timespec='seconds')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute('''
              INSERT INTO uploads (user_id, files, 
                                   analysis_skin_type, analysis_acne, analysis_rosacea, analysis_comedones,
                                   final_recommendation, upload_time)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_db_id, files_json_for_db,
                  analysis_results.get("skin_type"),
                  analysis_results.get("acne_status"),
                  analysis_results.get("rosacea_status"),
                  analysis_results.get("comedones_status"),
                  analysis_results.get("final_recommendation"),
                  now_iso))
            conn.commit()
            new_upload_id = c.lastrowid
        except sqlite3.Error as e:
            flash(f"Ошибка базы данных: {e}", "danger")
            conn.rollback()
            return redirect(request.url)
        finally:
            conn.close()
        
        flash('Анализ успешно завершен!', 'success')
        return redirect(url_for('analysis_detail', upload_id=new_upload_id))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT id, upload_time, final_recommendation 
        FROM uploads 
        WHERE user_id = ? 
        ORDER BY upload_time DESC
    ''', (user_db_id,))
    history_records = c.fetchall()
    conn.close()

    return render_template('dashboard.html', 
                           uploaded=False, 
                           history=history_records,
                           first_name=session.get('first_name', 'Пользователь'))

@app.route('/analysis/<int:upload_id>')
def analysis_detail(upload_id):
    if 'user_db_id' not in session:
        flash('Пожалуйста, войдите в систему.', 'warning')
        return redirect(url_for('index'))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT * FROM uploads WHERE id = ? AND user_id = ?
    ''', (upload_id, session['user_db_id']))
    analysis = c.fetchone()
    conn.close()

    if not analysis:
        flash('Анализ не найден или у вас нет к нему доступа.', 'danger')
        return redirect(url_for('dashboard'))
    
    image_files = []
    if analysis['files']:
        try:
            image_files = json.loads(analysis['files'])
        except json.JSONDecodeError:
            flash('Ошибка при чтении списка файлов анализа.', 'warning')
            # image_files останется пустым списком

    return render_template('analysis_detail.html', 
                           analysis=analysis, 
                           image_files=image_files,
                           first_name=session.get('first_name', 'Пользователь'))

@app.route('/photos/<filename>')
def uploaded_file(filename):
    if '..' in filename or filename.startswith('/'):
        return "Запрещено", 403
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logout')
def logout():
    session.clear()
    flash('Вы успешно вышли из системы.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): # Проверяем и создаем папку UPLOAD_FOLDER при старте
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(DB_PATH): # Создаем БД только если ее нет
        init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)