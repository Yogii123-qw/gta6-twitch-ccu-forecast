# Peramal penonton Twitch di hari rilis

Program ini meramalkan jumlah penonton bersamaan (concurrent viewers)
kategori Twitch sebuah gim di tiap jam ke-2 sampai ke-24 pada hari
rilisnya, hanya dari angka jam pertama, dalam lima kuantil
(0,05, 0,25, 0,5, 0,75, dan 0,95).

## Cara kerja, singkatnya

Bentuk kurva sepanjang hari tidak ditebak dari nol: ia dipinjam dari
peluncuran-peluncuran lampau yang disertakan di dalam kasus sebagai
pembanding.

1. **Angka jam pertama.** Diambil dari baris `known` untuk jam 1.
   Bila baris itu tidak memberi angka yang bisa dipakai, dipakai angka
   terakhir yang bisa dipakai dari `history`; bila tetap tidak ada,
   dipakai angka 1. Untuk semua perhitungan, angka ini dibatasi
   bawah di 1.
2. **Kurva pembanding.** Tiap pembanding disusun menjadi 24 angka jam
   hari rilisnya, dengan mengelompokkan baris per jamnya menurut
   selisih jam bulat dari awal hari pembanding itu. Kurva hanya sah
   bila ke-24 jamnya terisi angka positif dan angka jam pertamanya
   sekurang-kurangnya 5% dari angka terbesar kurva tersebut.
3. **Pemilihan donor.** Donor adalah kurva sah yang jam pertamanya
   dekat dengan angka jam pertama kasus ini dalam skala log (jarak
   tidak lebih dari 1,55). Bila yang lolos kurang dari enam, dipakai
   enam kurva terdekat. Donor diurutkan dengan kunci yang tetap
   (log angka jam pertama, lalu seluruh kurva), sehingga urutan
   pembanding di dalam berkas kasus tidak mengubah hasil.
4. **Rasio log.** Untuk tiap jam yang diminta, dihitung log rasio
   angka jam itu terhadap angka jam pertama tiap donor. Rata-rata
   rasio menjadi pusat kurva, dan akar varians populasinya (dengan
   batas bawah) menjadi sebarannya. Pusat dan tiap kurva geser
   kuantil lalu dihaluskan dengan rata-rata bergerak 3 titik yang
   berpusat dan sadar tepi.
5. **Kuantil.** Tiap kuantil adalah angka jam pertama dikali
   eksponen dari pusat ditambah geser kuantilnya, dibatasi bawah di
   1, dibulatkan ke 2 desimal (pecahan tepat setengah naik), dan
   dipaksa tidak menurun antar level.
6. **Jalan cadangan.** Bila tidak ada donor sama sekali, tiap jam
   diramal tetap di angka jam pertama, dengan sebaran 0,15 dikali
   akar dari (jam dikurangi 1).

Program ini deterministik: tanpa keacakan, tanpa jam sistem, tanpa
jaringan, dan hanya memakai standard library Python.

## Menjalankan

Program membaca tiap berkas `*.in` di direktori masukan dan menulis
satu berkas `*.out` dengan nama dasar yang sama di direktori
keluaran. Direktori masukan diambil dari `MAKEFUN_INPUT_DIR`
(bawaan `/input`), direktori keluaran dari `MAKEFUN_OUTPUT_DIR`
(bawaan `/output`).

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Tiap berkas keluaran berisi satu objek JSON ringkas, `{"hours": [...]}`,
dengan 23 baris untuk jam 2 sampai 24, masing-masing berisi nomor jam
dan lima angka kuantil. Python 3.13 sudah cukup; tidak ada paket
tambahan yang perlu dipasang.
