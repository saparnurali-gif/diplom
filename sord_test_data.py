import os
import librosa
import numpy as np
from scipy.io import wavfile

# Настройки путей к папкам (предполагаем структуру как в вашем проекте)
HUMAN_DIR = "R:\\Desktop\\осень семетр\\ОНИРС\\ПРОЕКТ\\data\\human\\human_test"          # Папка с human голосами
AI_DIR = "R:\\Desktop\\осень семетр\\ОНИРС\\ПРОЕКТ\\data\\ai\\ai_test"                # Папка с ai голосами
RE_RECORDED_DIR = "R:\\Desktop\\осень семетр\\ОНИРС\\ПРОЕКТ\\data\\re_recorded\\re_recorded_test"  # Папка с перезаписанными голосами
TEST_DATA_DIR = "R:\\Desktop\\осень семетр\\ОНИРС\\ПРОЕКТ\\test_data"       # Целевая папка

SAMPLE_RATE = 16000               # Частота дискретизации
DURATION = 3                      # Длина сегмента в секундах
MAX_TOTAL_FILES = 1389            # Максимальное общее количество файлов

# Вычисляем целевое количество сегментов на класс (примерно равное)
num_classes = 3
target_per_class = MAX_TOTAL_FILES // num_classes  # ~66 на класс

# Создаём целевую папку, если она не существует
if not os.path.exists(TEST_DATA_DIR):
    os.makedirs(TEST_DATA_DIR)
    print(f"Создана папка: {TEST_DATA_DIR}")

# Функция для обработки одной папки: разделение на сегменты и сохранение
def process_folder(source_dir, prefix, target_segments):
    file_list = [f for f in os.listdir(source_dir) if f.lower().endswith(".wav")]
    segment_counter = 0
    file_idx = 1  # Счётчик для нумерации файлов (human_001.wav, human_002.wav и т.д.)
    
    for file_name in file_list:
        if segment_counter >= target_segments:
            break
        
        source_path = os.path.join(source_dir, file_name)
        
        # Загружаем аудио
        audio, sr = librosa.load(source_path, sr=SAMPLE_RATE)
        total_samples = len(audio)
        segment_length = SAMPLE_RATE * DURATION
        num_segments = total_samples // segment_length
        
        if num_segments == 0:
            # Если короче 3 сек — растягиваем до 3 сек
            audio = librosa.util.fix_length(audio, size=segment_length)
            num_segments = 1
        
        # Делим на сегменты и сохраняем каждый как отдельный файл
        for i in range(num_segments):
            if segment_counter >= target_segments:
                break
            
            start = i * segment_length
            end = start + segment_length
            segment = audio[start:end]
            
            # Новое имя: prefix_001.wav, prefix_002.wav и т.д.
            new_name = f"{prefix}_{file_idx:03d}.wav"
            target_path = os.path.join(TEST_DATA_DIR, new_name)
            
            # Сохраняем сегмент как wav-файл
            wavfile.write(target_path, SAMPLE_RATE, segment)
            print(f"Сохранён сегмент {i+1} из {file_name} как {new_name}")
            
            segment_counter += 1
            file_idx += 1

    return segment_counter

# Обработка каждой папки с лимитом на сегменты
global_total = 0

print("\nОбработка human:")
human_count = process_folder(HUMAN_DIR, "human", target_per_class)
global_total += human_count

print("\nОбработка ai:")
ai_count = process_folder(AI_DIR, "ai", target_per_class)
global_total += ai_count

print("\nОбработка re_recorded:")
re_count = process_folder(RE_RECORDED_DIR, "re_recorded", target_per_class)
global_total += re_count

# Если общее меньше 100 (из-за нехватки данных), ничего не добавляем — это ок
print(f"\nГотово! Всего файлов в test_data: {global_total} (human: {human_count}, ai: {ai_count}, re_recorded: {re_count})")