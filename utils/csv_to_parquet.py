import pandas as pd
import yaml
import os


def convert_csv():
    # 1. Завантаження конфігурації
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = "data/EURUSD_M1_202511271905_202603091704.csv"
    save_path = config['paths']['data_path']

    if not os.path.exists(csv_path):
        print(f"❌ Файл не знайдено: {csv_path}. Експортуйте дані з MT5 (M1) та покладіть у папку data.")
        return

    print("⏳ Читання CSV та підбір кодування...")

    # Спроба прочитати файл з різними кодуваннями
    df = None
    encodings = ['utf-16', 'utf-16-le', 'utf-8', 'cp1251']

    for enc in encodings:
        try:
            # MT5 зазвичай використовує Tab (\t) як розділювач
            df = pd.read_csv(csv_path, sep='\t', encoding=enc, nrows=100)
            if len(df.columns) < 5:  # Якщо прочитало як одну колонку, пробуємо кому
                df = pd.read_csv(csv_path, sep=',', encoding=enc, nrows=100)

            # Якщо ми тут, значить файл прочитано успішно
            print(f"✅ Кодування знайдено: {enc}")
            df = pd.read_csv(csv_path, sep='\t' if len(df.columns) > 1 else ',', encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        print("❌ Не вдалося прочитати файл. Перевірте формат експорту в MT5.")
        return

    # 2. Очищення колонок
    # MT5 часто додає символи '<' '>' або пробіли в назви
    df.columns = [c.replace('<', '').replace('>', '').strip().upper() for c in df.columns]

    # Визначаємо колонки часу
    if 'DATE' in df.columns and 'TIME' in df.columns:
        df['time'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    elif 'TIME' in df.columns:  # Іноді дата і час в одній колонці
        df['time'] = pd.to_datetime(df['TIME'])
    else:
        # Якщо назви зовсім інші (наприклад, локалізовані), беремо за індексом
        df['time'] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1])

    # Залишаємо потрібні колонки та перейменовуємо
    # Стандартний порядок MT5: Date, Time, Open, High, Low, Close, TickVol, Vol, Spread
    # Нам потрібні OHLC + Volume (зазвичай це TICKVOL для форексу)
    mapping = {
        'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close',
        'TICKVOL': 'VOL', 'TICK_VOL': 'VOL', 'VOL': 'VOL'
    }

    df_new = pd.DataFrame()
    df_new['time'] = df['time']

    for mt5_col, our_col in mapping.items():
        if mt5_col in df.columns:
            df_new[our_col] = df[mt5_col]

    # Видаляємо дублікати та сортуємо
    df_new = df_new.drop_duplicates('time').sort_values('time').reset_index(drop=True)

    print(f"📊 Завантажено {len(df_new)} барів M1. Починаю resampling в M5...")

    # 3. Створення M5 з M1
    df_new.set_index('time', inplace=True)
    df_m5 = df_new.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'VOL': 'sum'
    }).dropna().reset_index()
    df_new.reset_index(inplace=True)

    # 4. Синхронізація (Merge)
    # Кожна хвилина M1 тепер "знає" параметри поточної 5-хвилинки
    df_final = pd.merge_asof(
        df_new,
        df_m5,
        on='time',
        direction='backward',
        suffixes=('', '_m5')
    )

    # 5. Збереження
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final.to_parquet(save_path, index=False)

    print("-" * 30)
    print(f"🚀 КОНВЕРТАЦІЯ ЗАВЕРШЕНА!")
    print(f"📁 Файл збережено: {save_path}")
    print(f"📈 Кількість рядків: {len(df_final)}")
    print(f"⏰ Перша свічка: {df_final['time'].min()}")
    print(f"⏰ Остання свічка: {df_final['time'].max()}")
    print("-" * 30)


if __name__ == "__main__":
    convert_csv()