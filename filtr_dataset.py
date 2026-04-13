import pandas as pd
from pathlib import Path
import shutil

# === Настройки ===
base = Path("data/human/cv-corpus-22.0-delta-2025-06-20-ru/")  # ← Укажи путь до корня Common Voice (где лежат validated.tsv)
clips_dir = base / "clips"         # Папка с исходными записями
validated_tsv = base / "validated.tsv"
output_dir = base / "validated_mp3"  # Папка для валидных записей
output_dir.mkdir(exist_ok=True)

# === 1. Загружаем таблицу ===
df = pd.read_csv(validated_tsv, sep="\t")

# === 2. Проверим, что есть нужная колонка ===
if "path" not in df.columns:
    raise ValueError("Файл validated.tsv не содержит колонку 'path' — проверь структуру файла")

# === 3. Считаем количество записей ===
print(f"Всего найдено {len(df)} валидных записей")

# === 4. Копируем валидные mp3 файлы ===
copied, skipped = 0, 0

for file_name in df["path"]:
    src = clips_dir / file_name
    dst = output_dir / file_name
    
    if src.exists():
        shutil.copy(src, dst)
        copied += 1
    else:
        skipped += 1

print(f"✅ Успешно скопировано: {copied}")
print(f"⚠️ Пропущено (файлы не найдены): {skipped}")
print(f"🎧 Все валидные записи теперь лежат в: {output_dir}")
