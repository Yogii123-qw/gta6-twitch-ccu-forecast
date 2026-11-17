"""Peramal penonton Twitch di hari rilis, dari jam pertama saja.

Gagasan utamanya: bentuk kurva sepanjang hari dipinjam dari peluncuran
lampau yang jam pertamanya mirip. Untuk tiap peluncuran pembanding,
diambil 24 angka jamnya di hari rilis. Pembanding yang jam pertamanya
dekat (dalam skala log) dengan jam pertama kasus ini dipakai sebagai
donor. Dari donor-donor itu dihitung, per jam, rata-rata dan sebaran
dari log rasio terhadap jam pertama masing-masing donor, lalu kedua
kurva itu dihaluskan dan dipakai untuk membentuk lima kuantil di
sekitar angka jam pertama kasus ini. Bila tidak ada donor yang layak,
ramalan jatuh ke bentuk datar dengan sebaran yang melebar mengikuti
akar waktu, seperti acuan dasar kompetisi.

Hanya standard library yang dipakai. Tidak ada keacakan, tidak ada
pembacaan jam sistem, tidak ada jaringan, dan urutan donor selalu
diurutkan dengan kunci yang sama, sehingga dua jalan pada kasus yang
sama selalu menulis berkas yang sama.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_lokasi_berkas_ini = Path(__file__).resolve()
_direktori_berkas_ini = str(_lokasi_berkas_ini.parent)
if _direktori_berkas_ini not in sys.path:
    sys.path.insert(0, _direktori_berkas_ini)

# Level kuantil yang diminta kasus, sesuai urutan di tiap baris keluaran.
LEVEL_YANG_DITAGIH = (0.05, 0.25, 0.5, 0.75, 0.95)
# Kuantil sebaran normal baku pada tiap level, dari yang paling rendah.
AMBANG_NORMAL = (
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722,
)
SELISIH_LEVEL_DIIZINKAN = 1e-9
JARAK_LOG_MAKSIMAL = 1.55
DONOR_PALING_SEDIKIT = 6
JAM_DALAM_SEHARI = 24
BATAS_BAWAH_VARIANS = 1e-9
SEBARAN_PER_AKAR_JAM = 0.15
RASIO_AWAL_MINIMAL = 0.05


def angka_yang_bisa_dipakai(nilai):
    """Ubah nilai menjadi bilangan positif, atau kembalikan None.

    Yang diterima hanya bilangan biasa dan string yang isinya bilangan.
    Boolean, nilai kosong, dan bilangan yang tidak hingga atau tidak
    positif ditolak, karena logaritmanya tidak terdefinisi.
    """
    if nilai is None or isinstance(nilai, bool):
        return None
    if isinstance(nilai, (int, float)):
        hasil = float(nilai)
    elif isinstance(nilai, str):
        tulisan = nilai.strip()
        if not tulisan:
            return None
        try:
            hasil = float(tulisan)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(hasil) or hasil <= 0:
        return None
    return hasil


def baca_waktu_iso(tulisan):
    """Ubah tulisan waktu ISO-8601 menjadi datetime, atau None bila rusak.

    Waktu tanpa keterangan zona dianggap waktu UTC, supaya baris yang
    hanya menulis tanggal dan jam tetap bisa dipakai.
    """
    if not isinstance(tulisan, str):
        return None
    potongan = tulisan.strip()
    if not potongan:
        return None
    if potongan[-1] in ("Z", "z"):
        potongan = potongan[:-1] + "+00:00"
    try:
        momen = datetime.fromisoformat(potongan)
    except ValueError:
        return None
    if momen.tzinfo is None:
        momen = momen.replace(tzinfo=timezone.utc)
    return momen


def nomor_jam_baris(baris):
    """Ambil nomor jam bulat dari sebuah baris, atau None bila tidak ada."""
    if not isinstance(baris, dict):
        return None
    mentah = baris.get("hour")
    if mentah is None or isinstance(mentah, bool):
        return None
    if isinstance(mentah, (int, float)):
        angka = float(mentah)
    elif isinstance(mentah, str):
        try:
            angka = float(mentah.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(angka):
        return None
    bulat = int(angka)
    if float(bulat) != angka:
        return None
    return bulat


def bulatkan_setengah_ke_atas(nilai):
    """Bulatkan ke 2 desimal, pecahan tepat setengah naik ke atas.

    Hasil yang kebetulan bulat dikembalikan sebagai bilangan bulat,
    supaya penulisannya di JSON sama seperti bilangan bulat biasa.
    """
    dalam_seratus = nilai * 100.0
    terbulat = math.floor(dalam_seratus + 0.5)
    kembali = terbulat / 100.0
    sebagai_bulat = int(kembali)
    if kembali == sebagai_bulat:
        return sebagai_bulat
    return kembali


def pangkat_terkendali(pangkat):
    """Eksponen dengan pangkat yang dibatasi, supaya tidak pernah meluap.

    Rasio donor dari data normal memberi pangkat puluhan saja, jauh
    dari batas ini, sehingga pembatasan tidak mengubah ramalan normal
    sama sekali. Ia hanya menjaga program dari crash bila kurva
    sampah membawa rasio ekstrem.
    """
    if pangkat > 700.0:
        pangkat = 700.0
    elif pangkat < -700.0:
        pangkat = -700.0
    return math.exp(pangkat)


def haluskan_tiga_titik(deret):
    """Rata-rata bergerak 3 titik yang berpusat, sadar tepi.

    Titik ujung hanya punya satu tetangga, jadi diratakan atas dua
    angka; titik tengah diratakan atas tiga angka.
    """
    banyak = len(deret)
    keluaran = []
    for posisi in range(banyak):
        mulai = posisi - 1
        if mulai < 0:
            mulai = 0
        selesai = posisi + 2
        if selesai > banyak:
            selesai = banyak
        jendela = deret[mulai:selesai]
        keluaran.append(sum(jendela) / len(jendela))
    return keluaran


def pastikan_level_sesuai(kasus, nama_berkas):
    """Gagalkan kasus bila daftar levelnya bukan lima nilai yang ditagih."""
    diminta = kasus.get("levels")
    if not isinstance(diminta, list) or len(diminta) != len(LEVEL_YANG_DITAGIH):
        raise ValueError(f"{nama_berkas} levelnya bukan lima nilai yang ditagih")
    for dapat, patokan in zip(diminta, LEVEL_YANG_DITAGIH):
        if dapat is None or isinstance(dapat, bool):
            raise ValueError(f"{nama_berkas} levelnya bukan lima nilai yang ditagih")
        try:
            if isinstance(dapat, str):
                angka = float(dapat.strip())
            else:
                angka = float(dapat)
        except (TypeError, ValueError):
            raise ValueError(f"{nama_berkas} levelnya bukan lima nilai yang ditagih")
        if not math.isfinite(angka) or abs(angka - patokan) > SELISIH_LEVEL_DIIZINKAN:
            raise ValueError(f"{nama_berkas} levelnya bukan lima nilai yang ditagih")


def cari_angka_jam_pertama(kasus):
    """Cari angka jam pertama: baris known jam 1, lalu history terakhir.

    Bila tidak satu pun baris memberi angka yang bisa dipakai, dipakai
    angka 1 sebagai jalan terakhir.
    """
    dikenal = kasus.get("known")
    if isinstance(dikenal, list):
        for baris in dikenal:
            if nomor_jam_baris(baris) != 1:
                continue
            angka = angka_yang_bisa_dipakai(baris.get("viewers"))
            if angka is not None:
                return angka
    riwayat = kasus.get("history")
    if isinstance(riwayat, list):
        for baris in reversed(riwayat):
            if not isinstance(baris, dict):
                continue
            angka = angka_yang_bisa_dipakai(baris.get("viewers"))
            if angka is not None:
                return angka
    return 1.0


def susun_kurva_harian(pembanding):
    """Susun 24 angka jam hari rilis dari satu peluncuran pembanding.

    Baris dikelompokkan ke ember jam menurut selisih jam bulat (ke
    bawah) dari awal hari pembanding; ember 0 adalah jam pertama.
    Baris yang waktunya atau angkanya rusak dilewati. Kurva dianggap
    sah hanya bila ke-24 ember terisi dan angka jam pertamanya tidak
    kurang dari 5% angka terbesar kurva itu.
    """
    if not isinstance(pembanding, dict):
        return None
    awal_hari = baca_waktu_iso(pembanding.get("day_start"))
    if awal_hari is None:
        return None
    per_jam = pembanding.get("hourly")
    if not isinstance(per_jam, list):
        return None
    ember = [None] * JAM_DALAM_SEHARI
    langkah = timedelta(hours=1)
    for baris in per_jam:
        if not isinstance(baris, dict):
            continue
        momen = baca_waktu_iso(baris.get("from"))
        if momen is None:
            continue
        angka = angka_yang_bisa_dipakai(baris.get("viewers"))
        if angka is None:
            continue
        posisi = (momen - awal_hari) // langkah
        if 0 <= posisi < JAM_DALAM_SEHARI and ember[posisi] is None:
            ember[posisi] = angka
    if any(slot is None for slot in ember):
        return None
    kurva = [float(slot) for slot in ember]
    if kurva[0] / max(kurva) < RASIO_AWAL_MINIMAL:
        return None
    return kurva


def tentukan_donor(semua_kurva, patokan):
    """Pilih donor dari kurva-kurva sah, dengan urutan yang tetap.

    Kurva yang jarak log jam pertamanya dari patokan tidak melebihi
    batas dipakai semua. Bila yang lolos kurang dari enam, dipakai
    enam kurva terdekat, dengan seri jarak dipecah oleh kunci urut
    yang sama. Donor terpilih lalu diurutkan berdasarkan log angka
    jam pertama, kemudian seluruh isi kurva, sebelum dipakai, supaya
    urutan pembanding di dalam kasus tidak mengubah hasil.
    """
    if not semua_kurva:
        return []
    log_patokan = math.log(patokan)
    calon = []
    for kurva in semua_kurva:
        log_awal = math.log(kurva[0])
        jarak = abs(log_awal - log_patokan)
        calon.append((jarak, log_awal, tuple(kurva), kurva))
    lolos = [butir for butir in calon if butir[0] <= JARAK_LOG_MAKSIMAL]
    if len(lolos) >= DONOR_PALING_SEDIKIT:
        terpilih = lolos
    else:
        berdasarkan_jarak = sorted(calon, key=lambda butir: (butir[0], butir[1], butir[2]))
        terpilih = berdasarkan_jarak[:DONOR_PALING_SEDIKIT]
    terpilih.sort(key=lambda butir: (butir[1], butir[2]))
    return [butir[3] for butir in terpilih]


def baris_ramalan_datar(patokan, jam):
    """Ramalan datar untuk satu jam: tetap di patokan, sebaran melebar."""
    jarak_dari_awal = jam - 1
    if jarak_dari_awal < 0:
        jarak_dari_awal = 0
    sebaran = SEBARAN_PER_AKAR_JAM * math.sqrt(jarak_dari_awal)
    kuantil = []
    for ambang in AMBANG_NORMAL:
        mentah = patokan * pangkat_terkendali(ambang * sebaran)
        kuantil.append(bulatkan_setengah_ke_atas(mentah))
    return {"hour": jam, "q": kuantil}


def ramalkan_satu_kasus(kasus, nama_berkas):
    """Bentuk ramalan lengkap untuk satu kasus."""
    if not isinstance(kasus, dict):
        raise ValueError(f"{nama_berkas} bukan objek kasus")
    pastikan_level_sesuai(kasus, nama_berkas)
    angka_awal = cari_angka_jam_pertama(kasus)
    patokan = angka_awal if angka_awal >= 1.0 else 1.0

    jam_yang_diminta = []
    daftar_ramal = kasus.get("forecast")
    if isinstance(daftar_ramal, list):
        for baris in daftar_ramal:
            jam = nomor_jam_baris(baris)
            if jam is not None:
                jam_yang_diminta.append(jam)
    jam_yang_diminta.sort()

    kurva_sah = []
    daftar_pembanding = kasus.get("comparables")
    if isinstance(daftar_pembanding, list):
        for pembanding in daftar_pembanding:
            kurva = susun_kurva_harian(pembanding)
            if kurva is not None:
                kurva_sah.append(kurva)
    donor = tentukan_donor(kurva_sah, patokan)

    if not donor:
        baris = [baris_ramalan_datar(patokan, jam) for jam in jam_yang_diminta]
        return {"hours": baris}

    banyak_donor = len(donor)
    daftar_tengah = []
    daftar_sebar = []
    for jam in jam_yang_diminta:
        posisi = jam - 1
        if 0 <= posisi < JAM_DALAM_SEHARI:
            rasio_log = []
            for kurva in donor:
                rasio_log.append(math.log(kurva[posisi] / kurva[0]))
            tengah = sum(rasio_log) / banyak_donor
            jumlah_kuadrat = sum((nilai - tengah) ** 2 for nilai in rasio_log)
            varians = jumlah_kuadrat / banyak_donor
            if varians < BATAS_BAWAH_VARIANS:
                varians = BATAS_BAWAH_VARIANS
            daftar_tengah.append(tengah)
            daftar_sebar.append(math.sqrt(varians))
        else:
            # Jam di luar jangkauan kurva harian tidak punya rasio donor;
            # perlakukan sebagai jam datar supaya barisnya tetap terisi.
            daftar_tengah.append(0.0)
            jarak_dari_awal = jam - 1
            if jarak_dari_awal < 0:
                jarak_dari_awal = 0
            daftar_sebar.append(SEBARAN_PER_AKAR_JAM * math.sqrt(jarak_dari_awal))

    tengah_halus = haluskan_tiga_titik(daftar_tengah)
    geser_halus = []
    for ambang in AMBANG_NORMAL:
        geser_mentah = [ambang * sebar for sebar in daftar_sebar]
        geser_halus.append(haluskan_tiga_titik(geser_mentah))

    baris = []
    for urutan, jam in enumerate(jam_yang_diminta):
        kuantil = []
        for tingkat in range(len(AMBANG_NORMAL)):
            mentah = patokan * pangkat_terkendali(tengah_halus[urutan] + geser_halus[tingkat][urutan])
            if mentah < 1.0:
                mentah = 1.0
            kuantil.append(bulatkan_setengah_ke_atas(mentah))
        for tingkat in range(1, len(kuantil)):
            if kuantil[tingkat] < kuantil[tingkat - 1]:
                kuantil[tingkat] = kuantil[tingkat - 1]
        baris.append({"hour": jam, "q": kuantil})
    return {"hours": baris}


def jalankan():
    """Proses tiap berkas kasus di direktori masukan, tulis keluarannya."""
    direktori_masuk = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    direktori_keluar = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    direktori_keluar.mkdir(parents=True, exist_ok=True)
    berkas_kasus = sorted(direktori_masuk.glob("*.in"), key=lambda berkas: berkas.name)
    for berkas in berkas_kasus:
        kasus = json.loads(berkas.read_text(encoding="utf-8"))
        hasil = ramalkan_satu_kasus(kasus, berkas.name)
        tulisan = json.dumps(hasil, separators=(",", ":")) + "\n"
        tujuan = direktori_keluar / (berkas.stem + ".out")
        tujuan.write_text(tulisan, encoding="utf-8")


if __name__ == "__main__":
    jalankan()
