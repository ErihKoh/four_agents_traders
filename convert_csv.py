import pandas as pd
import os


INPUT_FILE = "data/EURUSD_M5_202408010000_202603091940.csv"
OUTPUT_FILE = "data/eurusd_m5.parquet"


def main():
    print("📂 Завантаження CSV...")

    # Спробуємо Tab, потім кому як роздільник
    try:
        df = pd.read_csv(INPUT_FILE, sep='\t')
        if len(df.columns) < 5:
            df = pd.read_csv(INPUT_FILE, sep=',')
    except Exception as e:
        print(f"❌ Помилка читання файлу: {e}")
        return

    print(f"✅ Завантажено {len(df)} рядків")
    print(f"   Колонки (оригінал): {df.columns.tolist()}")

    # Прибираємо кутові дужки з назв колонок та переводимо в нижній регістр
    df.columns = [c.strip().replace('<', '').replace('>', '').lower() for c in df.columns]
    print(f"   Колонки (очищені): {df.columns.tolist()}")

    # Об'єднуємо DATE + TIME в одну колонку 'time'
    if 'date' in df.columns and 'time' in df.columns:
        df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed')
        df = df.drop(columns=['date'])
    elif 'datetime' in df.columns:
        df['time'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.drop(columns=['datetime'])
    else:
        print("❌ Не знайдено колонки з датою/часом!")
        return

    # Перейменовуємо tickvol → vol якщо потрібно
    if 'tickvol' in df.columns and 'vol' not in df.columns:
        df = df.rename(columns={'tickvol': 'vol'})

    # Прибираємо непотрібні колонки
    df = df.drop(columns=['spread', 'vol'], errors='ignore')

    # Залишаємо тільки потрібні колонки M1
    required = ['time', 'open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Відсутні колонки: {missing}")
        print(f"   Доступні: {df.columns.tolist()}")
        return

    df = df[required].copy()

    # Сортуємо по часу
    df = df.sort_values('time').reset_index(drop=True)

    # Перевірка на дублікати
    dupes = df.duplicated(subset=['time']).sum()
    if dupes > 0:
        print(f"⚠️  Знайдено {dupes} дублікатів — видаляємо")
        df = df.drop_duplicates(subset=['time']).reset_index(drop=True)

    print(f"\n📊 M1 дані: {len(df)} свічок")
    print(f"   Період: {df['time'].min()} → {df['time'].max()}")

    # =============================================
    # РЕСЕМПЛІНГ M1 → M5 (БЕЗ ВИТОКУ ДАНИХ)
    # =============================================
    print("\n🔄 Генерація M5 з M1...")

    df_m5 = df.set_index('time').resample('5min').agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
    }).dropna()

    # КРИТИЧНИЙ ФІКСl: зсуваємо мітку часу M5 на +5 хвилин
    # Це означає: свічка 21:40-21:44 отримує мітку 21:45
    # merge_asof НЕ використає її раніше ніж настане 21:45 → нема витоку!
    df_m5.index = df_m5.index + pd.Timedelta(minutes=5)
    df_m5 = df_m5.reset_index()

    # Перейменовуємо колонки M5
    df_m5.columns = ['time'] + [f"{c}_m5" for c in df_m5.columns if c != 'time']

    print(f"   M5 свічок: {len(df_m5)}")

    # =============================================
    # POINT-IN-TIME JOIN M1 + M5
    # =============================================
    df_combined = pd.merge_asof(
        df,
        df_m5,
        on='time',
        direction='backward'
    )

    before = len(df_combined)
    df_combined = df_combined.dropna().reset_index(drop=True)
    after = len(df_combined)

    if before != after:
        print(f"⚠️  Видалено {before - after} рядків з NaN після join")

    # =============================================
    # ЗБЕРЕЖЕННЯ
    # =============================================
    os.makedirs("data", exist_ok=True)
    df_combined.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n✅ Збережено: {OUTPUT_FILE}")
    print(f"   Рядків: {len(df_combined)}")
    print(f"   Колонки: {df_combined.columns.tolist()}")
    print(f"   Період: {df_combined['time'].min()} → {df_combined['time'].max()}")
    print(f"\n   Перші 3 рядки:")
    print(df_combined.head(3).to_string())


if __name__ == "__main__":
    main()