import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

path = r"R:\Desktop\осень семетр\ОНИРС\ПРОЕКТ\data\ai\68ff92b6486d92066ee226c6_result.wav"

# Загружаем аудио
audio, sr = librosa.load(path, sr=None)  # sr=None = оригинальная частота
print(f"Длина: {len(audio)/sr:.2f} сек, Частота дискретизации: {sr} Гц")

# Строим спектрограмму
spec = librosa.feature.melspectrogram(y=audio, sr=sr)
spec_db = librosa.power_to_db(spec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.title("Mel-спектрограмма")
plt.colorbar(format='%+2.0f dB')
plt.show()
plt.savefig("spectrogram.png")
print("✅ Спектрограмма сохранена как 'spectrogram.png'")
