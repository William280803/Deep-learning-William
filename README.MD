## Dependensi atau alat dan pustaka yang digunakan

- [Python 3](https://www.python.org/downloads/release/python-380/), [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
- Langkah awalan utuk menjalankan program ketikan perintah berikut di dalam terminal
  `pip install -r requirements.txt`.

## Penggunaan Dasar

Repositori saat ini kompatibel dengan `tensorflow` dan menggunakan API Keras menggunakan `tensorflow.keras` pustaka.

- Download the FER-2013 dataset from [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view?usp=sharing) and unzip it inside the `src` folder. This will create the folder `data`.

- Jika ingin melatih kumpulan data (model), ketikan:

```bash
python main.py --mode train
```

- Jika ingin melihat prediksi/hasil tanpa pelatihan lagi, Anda dapat mendownload model terlatih dari sini dan kemudian menjalankan:

```bash
python main.py --mode display
```

- Struktur Folder:  
  src:

  - data (folder)
  - `main.py` (file)
  - `dataset_prepare.py` (file)
  - `haarcascade_frontalface_default.xml` (file)
  - `model.h5` (file)
  - `requirements.txt` (file)

- Penerapan ini secara default mendeteksi emosi pada semua wajah di umpan webcam. Dengan CNN 4-lapis sederhana, akurasi pengujian mencapai 95% dalam 100 epoch.

## Persiapan Data (opsional jika ingin melatih data dengan kumpulan data sendiri)

- The [dataset FER2013 asli dalam Kaggle](https://www.kaggle.com/deadskull7/fer2013) tersedia sebagai file csv tunggal. Saya telah mengonversi menjadi kumpulan data gambar dalam format PNG untuk pelatihan / pengujian dan memberikan ini sebagai kumpulan data di bagian sebelumnya.

- ika Anda ingin bereksperimen dengan kumpulan data baru, Anda mungkin harus berurusan dengan data dalam format csv. Saya telah memberikan kode yang saya tulis untuk preprocessing data di `dataset_prepare.py` file yang dapat digunakan untuk referensi.

## Algoritma

- Pertama, metode **haar cascade** digunakan untuk mendeteksi wajah di setiap bingkai dari umpan webcam.

- Wilayah gambar yang berisi wajah diubah ukurannya menjadi **48x48** dan diteruskan sebagai input ke CNN.

- Jaringan mengeluarkan daftar **skor softmax** untuk tujuh kelas emosi.

- Emosi dengan skor maksimal ditampilkan di layar.
