
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import io
from typing import Tuple, Dict
import re

# =======================
# Utils
# =======================
def convert_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    new_columns = []
    for col in df.columns:
        snake = re.sub(r'[^a-zA-Z0-9]', '_', str(col).lower())
        snake = re.sub(r'_+', '_', snake).strip('_')
        new_columns.append(snake)
    df.columns = new_columns
    return df

_ddmmyyyy_pat = re.compile(r'^\s*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\s*$')

def _parse_one_tanggal(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.to_datetime(v, errors='coerce')
    s = str(v).strip()
    if s == "" or s in {"-", "nan", "NaT"}:
        return pd.NaT
    if _ddmmyyyy_pat.match(s):
        out = pd.to_datetime(s, errors='coerce', dayfirst=True)
        if out is not None:
            return out
    if re.fullmatch(r'^\d+(\.\d+)?$', s):
        try:
            return pd.to_datetime(float(s), unit='D', origin='1899-12-30', errors='coerce')
        except Exception:
            pass
    out = pd.to_datetime(s, errors='coerce', dayfirst=True)
    if pd.isna(out):
        out = pd.to_datetime(s, errors='coerce')
    return out

def parse_tanggal_series(series_like) -> pd.Series:
    return pd.Series([_parse_one_tanggal(v) for v in series_like],
                     index=getattr(series_like, 'index', None))

def _is_missing_scan(val) -> bool:
    if pd.isna(val): return True
    s = str(val).strip().lower()
    return s in {"", "-", "0", "00:00", "00:00:00"}

def _safe_to_minutes(x) -> int:
    if pd.isna(x) or x == '': return 0
    s = str(x)
    try:
        if ':' in s:
            hh, mm, *_ = (s.split(':') + ['0','0'])[:2]
            return int(float(hh))*60 + int(float(mm))
        return int(float(s))
    except Exception:
        return 0

# =======================
# Mapping Jadwal
# =======================
JADWAL_MAP = {
    "libur rutin": "LIBUR",
    "libur": "LIBUR",
    "izin dinas": "IZIN_DINAS",
    "izin keperluan kantor": "IZIN_DINAS",
    "dinas": "IZIN_DINAS",
    "tidak hadir": "TIDAK_HADIR",
    "alpa": "TIDAK_HADIR",
    "absen": "TIDAK_HADIR",
    "tanpa keterangan": "TIDAK_HADIR",
    "izin pribadi": "IZIN_PRIBADI",
    "izin": "IZIN_PRIBADI",
    "cuti alasan pribadi": "IZIN_PRIBADI",
    "sakit tanpa skd": "SAKIT_TANPA_SKD",
    "sakit tdk skd": "SAKIT_TANPA_SKD",
    "sakit tanpa surat dokter": "SAKIT_TANPA_SKD",
    "sakit non skd": "SAKIT_TANPA_SKD",
    "sakit ada skd": "SAKIT_SKD",
    "sakit skd": "SAKIT_SKD",
    "sakit dengan skd": "SAKIT_SKD",
    "sakit surat dokter": "SAKIT_SKD",
}
JADWAL_KEYS = sorted(JADWAL_MAP.keys(), key=len, reverse=True)

def normalize_jadwal_value(val: str) -> str:
    if pd.isna(val): return ""
    s = str(val).strip().lower()
    for k in JADWAL_KEYS:
        if k in s:
            return JADWAL_MAP[k]
    return ""

# =======================
# Loader (cached & ringan)
# =======================
@st.cache_data(show_spinner=False)
def _parse_uploaded_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import io as _io
    bio = _io.BytesIO(file_bytes)

    def _read_csv(_bio, header_try):
        _bio.seek(0)
        try:
            return pd.read_csv(_bio, header=header_try, converters={'tanggal': lambda x: x}, encoding='utf-8-sig')
        except Exception:
            _bio.seek(0)
            return pd.read_csv(_bio, header=header_try, converters={'tanggal': lambda x: x}, encoding='latin1')

    if filename.lower().endswith('.csv'):
        df = _read_csv(bio, header_try=1)
        if df.shape[1] <= 1:
            df = _read_csv(bio, header_try=0)
    else:
        df = pd.read_excel(bio, header=1, converters={'tanggal': lambda x: x})
        if df.shape[1] <= 1:
            bio.seek(0)
            df = pd.read_excel(bio, header=0, converters={'tanggal': lambda x: x})

    df = convert_to_snake_case(df)

    if 'tanggal' in df.columns:
        df['tanggal'] = parse_tanggal_series(df['tanggal'])

    if 'jadwal' in df.columns:
        df['jadwal_kategori'] = df['jadwal'].map(normalize_jadwal_value).fillna("")
    else:
        df['jadwal_kategori'] = ""

    # Jangan baca scan pada hari sakit (dengan/tanpa SKD)
    if 'jadwal_kategori' in df.columns:
        mask_sakit = df['jadwal_kategori'].astype(str).str.upper().isin(['SAKIT_SKD', 'SAKIT_TANPA_SKD'])
        if mask_sakit.any():
            df.loc[mask_sakit, ['scan_masuk', 'scan_pulang']] = np.nan
            if 'terlambat_menit' in df.columns:
                df.loc[mask_sakit, 'terlambat_menit'] = 0

    aliases_in  = ["scan_masuk","scan_in","masuk","in"]
    aliases_out = ["scan_pulang","scan_out","pulang","out"]
    def _first_col(cols):
        for c in cols:
            if c in df.columns: return c
        return None
    col_in = _first_col(aliases_in)
    col_out = _first_col(aliases_out)
    if col_in and col_in != "scan_masuk": df["scan_masuk"] = df[col_in]
    elif "scan_masuk" not in df.columns: df["scan_masuk"] = np.nan
    if col_out and col_out != "scan_pulang": df["scan_pulang"] = df[col_out]
    elif "scan_pulang" not in df.columns: df["scan_pulang"] = np.nan

    is_libur_flag = pd.Series(False, index=df.index)
    if 'jam_kerja' in df.columns:
        jk = df['jam_kerja'].astype(str).str.lower()
        is_libur_flag |= jk.str.contains('libur rutin', regex=False) | jk.str.contains('libur', regex=False)
    if 'jadwal' in df.columns:
        jdl = df['jadwal'].astype(str).str.lower()
        is_libur_flag |= (
            jdl.str.contains('libur rutin', regex=False) |
            jdl.str.contains('libur', regex=False) |
            jdl.str.contains('izin dinas', regex=False) |
            jdl.str.contains('izin keperluan kantor', regex=False)
        )
    df['is_libur'] = is_libur_flag.fillna(False)

    df['terlambat_menit'] = df['terlambat'].map(_safe_to_minutes).astype(int) if 'terlambat' in df.columns else 0

    if 'departemen' not in df.columns:
        for alt in ['department','departement','dept','divisi','bagian','unit']:
            if alt in df.columns:
                df['departemen'] = df[alt]; break
    if 'departemen' not in df.columns:
        df['departemen'] = ""

    return df

def load_data(uploaded_file) -> pd.DataFrame:
    try:
        file_bytes = uploaded_file.getvalue()
        return _parse_uploaded_bytes(file_bytes, uploaded_file.name)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# =======================
# Business rules
# =======================
def is_cleaning_service(jabatan: str) -> bool:
    """
    Deteksi Cleaning Service berbasis JABATAN saja.
    Menangani variasi: 'Cleaning Service', 'Cleaning Services',
    'cleaning-service', 'CS', 'petugas cleaning', dll.
    """
    if pd.isna(jabatan):
        return False

    s = str(jabatan).strip().lower()
    # Normalisasi spasi & hilangkan tanda baca agar match lebih robust
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    patterns = [
        r'\bcleaning\s*service(s)?\b',  # cleaning service / services
        r'\bcleaning\s*staff\b',
        r'\bpetugas\s*cleaning\b',
        r'\bcs\b',                      # singkatan
        r'\boffice\s*boy\b', r'\bob\b'  # opsional jika di tempat Anda dianggap bagian CS
    ]
    return any(re.search(p, s) for p in patterns)

def hitung_potongan_terlambat_perkejadian(minutes: int) -> int:
    if minutes <= 0: return 0
    elif minutes <= 5: return minutes * 1000
    elif minutes <= 10: return 10000
    elif minutes <= 15: return 15000
    else: return 20000

def hitung_potongan_absen_cleaning_service(tidak_hadir: int, izin_pribadi_full: int,
                                           sakit_tanpa_skd: int, gaji_per_hari: float) -> Tuple[int, str]:
    total_pot = 0; detail=[]
    if sakit_tanpa_skd > 0: total_pot += sakit_tanpa_skd * gaji_per_hari; detail.append(f"Sakit tanpa SKD: {sakit_tanpa_skd} hari")
    if tidak_hadir > 0: total_pot += tidak_hadir * gaji_per_hari; detail.append(f"Tidak hadir: {tidak_hadir} hari")
    if izin_pribadi_full > 0: total_pot += izin_pribadi_full * gaji_per_hari; detail.append(f"Izin pribadi (full): {izin_pribadi_full} hari")
    return int(total_pot), "; ".join(detail)

def hitung_potongan_absen_non_cleaning_service(tidak_hadir: int, izin_pribadi_full: int,
                                               sakit_tanpa_skd: int, gaji_per_hari: float) -> Tuple[int, str]:
    total_pot = 0; detail=[]
    if tidak_hadir > 0: total_pot += tidak_hadir * gaji_per_hari; detail.append(f"Tidak hadir: {tidak_hadir} hari")
    if izin_pribadi_full > 0: total_pot += izin_pribadi_full * gaji_per_hari; detail.append(f"Izin pribadi (full): {izin_pribadi_full} hari")
    if sakit_tanpa_skd > 2:
        hari_dipotong = sakit_tanpa_skd - 2
        total_pot += hari_dipotong * gaji_per_hari
        detail.append(f"Sakit tanpa SKD (di atas toleransi): {hari_dipotong} hari")
    return int(total_pot), "; ".join(detail)

# =======================
# Core processing
# =======================
def process_payroll_data(
    scanlog_df: pd.DataFrame,
    gaji_per_hari_map: Dict[str, float],
    toleransi_telat_kejadian: int = 3,
    izin_pribadi_marks: pd.DataFrame = None
) -> pd.DataFrame:

    if scanlog_df is None or scanlog_df.empty:
        return pd.DataFrame()

    for col, default in [
        ("nip", ""), ("nama", ""), ("jabatan", ""), ("departemen", ""),
        ("jadwal_kategori", ""), ("is_libur", False),
        ("terlambat_menit", 0), ("scan_masuk", np.nan), ("scan_pulang", np.nan),
        ("tanggal", pd.NaT),
    ]:
        if col not in scanlog_df.columns:
            scanlog_df[col] = default

    if not np.issubdtype(scanlog_df["tanggal"].dtype, np.datetime64):
        scanlog_df["tanggal"] = parse_tanggal_series(scanlog_df["tanggal"])

    izin_mark = pd.DataFrame(columns=['nip','tanggal','tandai_ijin','durasi_ijin'])
    if isinstance(izin_pribadi_marks, pd.DataFrame) and not izin_pribadi_marks.empty:
        izin_mark = izin_pribadi_marks.copy()
        if not np.issubdtype(izin_mark["tanggal"].dtype, np.datetime64):
            izin_mark["tanggal"] = parse_tanggal_series(izin_mark["tanggal"])

    results = []

    group_keys = ['nip', 'nama', 'jabatan', 'departemen'] if 'departemen' in scanlog_df.columns else ['nip','nama','jabatan']
    for _, group in scanlog_df.sort_values('tanggal').groupby(group_keys, dropna=False):
        nip = str(group.iloc[0]['nip'])
        nama = group.iloc[0]['nama']
        jabatan = group.iloc[0]['jabatan']
        departemen = group.iloc[0]['departemen'] if 'departemen' in group.columns else ""
        is_cs = is_cleaning_service(jabatan)

        if group["tanggal"].notna().any():
            tmin_ts = pd.to_datetime(group["tanggal"]).min()
            tmax_ts = pd.to_datetime(group["tanggal"]).max()
            periode_tanggal = f"{tmin_ts.strftime('%d-%m-%y')} – {tmax_ts.strftime('%d-%m-%y')}"
        else:
            periode_tanggal = ""

        kerja_base = (~group['is_libur']) & (~group['jadwal_kategori'].isin(["LIBUR","IZIN_DINAS"]))
        not_full_absen = (~group['jadwal_kategori'].isin(["TIDAK_HADIR","SAKIT_TANPA_SKD","SAKIT_SKD","IZIN_PRIBADI"]))
        ada_scan = (~group['scan_masuk'].apply(_is_missing_scan)) | (~group['scan_pulang'].apply(_is_missing_scan))
        hadir_mask = kerja_base & not_full_absen & ada_scan

        jumlah_hari_hadir = int(hadir_mask.sum())

        izin_parsial_dates = set()
        if isinstance(izin_mark, pd.DataFrame) and not izin_mark.empty:
            mk_emp = izin_mark[(izin_mark['nip'].astype(str) == nip) & (izin_mark['tandai_ijin'] == True)]
            if not mk_emp.empty:
                valid_dates = set(pd.to_datetime(group['tanggal'].dropna()).dt.date.tolist())
                for _, r in mk_emp.iterrows():
                    tgl = r['tanggal']
                    if pd.isna(tgl):
                        continue
                    d = pd.to_datetime(tgl).date()
                    if d in valid_dates:
                        izin_parsial_dates.add(d)

        mask_izin_parsial_hari = group['tanggal'].apply(lambda x: (not pd.isna(x)) and (x.date() in izin_parsial_dates))

        miss_in_on_hadir = int((hadir_mask & group['scan_masuk'].apply(_is_missing_scan)).sum())

        late_series = group.loc[hadir_mask & (~mask_izin_parsial_hari), 'terlambat_menit'].astype(int).tolist()
        late_positive = [m for m in late_series if m > 0]
        total_telat_positif = len(late_positive)

        tol_awal = max(0, int(toleransi_telat_kejadian))
        tol_terpakai_missin = min(tol_awal, max(0, miss_in_on_hadir))
        tol_sisa = max(0, tol_awal - miss_in_on_hadir)

        if tol_sisa >= total_telat_positif:
            charged_minutes = []
        else:
            charged_minutes = late_positive[tol_sisa:]

        potongan_terlambat = sum(hitung_potongan_terlambat_perkejadian(m) for m in charged_minutes)
        telat_dipotong = len(charged_minutes)

        kat = group.get("jadwal_kategori", pd.Series([], dtype=str)).fillna("")
        tidak_hadir = int((kat == "TIDAK_HADIR").sum())
        sakit_tanpa_skd = int((kat == "SAKIT_TANPA_SKD").sum())
        sakit_skd = int((kat == "SAKIT_SKD").sum())
        izin_pribadi_full = int((kat == "IZIN_PRIBADI").sum())

        gph = float(gaji_per_hari_map.get(nip, 0) or 0.0)
        if is_cs:
            potongan_absen, _ = hitung_potongan_absen_cleaning_service(
                tidak_hadir, izin_pribadi_full, sakit_tanpa_skd, gph
            )
        else:
            potongan_absen, _ = hitung_potongan_absen_non_cleaning_service(
                tidak_hadir, izin_pribadi_full, sakit_tanpa_skd, gph
            )

        potongan_izin_parsial = 0; izin_parsial_cnt = 0
        if isinstance(izin_mark, pd.DataFrame) and not izin_mark.empty:
            mk = izin_mark[(izin_mark['nip'].astype(str) == nip) & (izin_mark['tandai_ijin'] == True)]
            if not mk.empty:
                valid_dates = set(pd.to_datetime(group['tanggal'].dropna()).dt.date.tolist())
                for _, r in mk.iterrows():
                    tgl = r['tanggal']
                    if pd.isna(tgl) or pd.to_datetime(tgl).date() not in valid_dates:
                        continue
                    dur = str(r.get('durasi_ijin', '')).strip().lower()
                    if dur in ('0-2 jam','0–2 jam','0 - 2 jam'):
                        pot = 10000
                    elif dur in ('2-3 jam','2–3 jam','2 - 3 jam'):
                        pot = 0.30 * gph
                    elif dur in ('3-4 jam','3–4 jam','3 - 4 jam'):
                        pot = 0.40 * gph
                    elif dur in ('4-5 jam','4–5 jam','4 - 5 jam'):
                        pot = 0.50 * gph
                    else:
                        pot = 0
                    if pot > 0:
                        izin_parsial_cnt += 1
                    potongan_izin_parsial += pot
        potongan_izin_parsial = int(round(potongan_izin_parsial, 0))

        gaji_bruto = int(round(gph * jumlah_hari_hadir, 0))

        total_potongan = int(potongan_terlambat) + int(potongan_absen) + int(potongan_izin_parsial)
        gaji_akhir = int(gaji_bruto - total_potongan)

        alasan_parts = []
        alasan_parts.append(f"Hari hadir: {jumlah_hari_hadir}")
        if miss_in_on_hadir > 0:
            alasan_parts.append(f"Tidak scan MASUK saat hadir: {miss_in_on_hadir} hari (toleransi berkurang {tol_terpakai_missin}x)")
        if total_telat_positif > 0:
            alasan_parts.append(f"Telat dihitung (setelah toleransi kejadian): {telat_dipotong}x dari {total_telat_positif}x")
        if potongan_absen > 0:
            alasan_parts.append(f"Potongan absen: Rp {potongan_absen:,}")
        if izin_parsial_cnt > 0:
            alasan_parts.append(f"Izin pribadi parsial: {izin_parsial_cnt} kejadian (Rp {potongan_izin_parsial:,})")
        if sakit_skd > 0:
            alasan_parts.append(f"Sakit dgn SKD: {sakit_skd} hari")
        if "is_libur" in group.columns and group["is_libur"].any():
            lib = int((group['is_libur']).sum())
            if lib > 0:
                alasan_parts.append(f"Info: Libur/izin dinas tidak dihitung denda ({lib} hari info)")

        results.append({
            "nip": nip, "nama": nama, "jabatan": jabatan,
            "departemen": departemen,
            "periode_tanggal": periode_tanggal,
            "gaji_per_hari": int(gph),
            "hari_hadir": int(jumlah_hari_hadir),
            "toleransi_telat_awal": int(tol_awal),
            "toleransi_telat_terpakai_miss_in": int(tol_terpakai_missin),
            "toleransi_telat_sisa": int(tol_sisa),
            "total_telat_kejadian": int(total_telat_positif),
            "telat_dipotong": int(telat_dipotong),
            "potongan_terlambat": int(potongan_terlambat),
            "potongan_absen": int(potongan_absen),
            "potongan_izin_pribadi_parsial": int(potongan_izin_parsial),
            "total_potongan": int(total_potongan),
            "gaji_bruto": int(gaji_bruto),
            "gaji_akhir": int(gaji_akhir),
            "alasan_potongan": "; ".join(alasan_parts) if alasan_parts else "Tidak ada potongan"
        })

    return pd.DataFrame(results)

# =======================
# Fitur Karyawan Rajin
# =======================
def build_rajin_recap(
    scan_df: pd.DataFrame,
    toleransi_rajin_menit: int = 0,
    izin_marks_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghasilkan:
      - rekap_rajin: jumlah_hari_kerja, jumlah_bersih, status
      - detail_tidak_rajin: daftar per karyawan berisi 'alasan_tidak_rajin' dan 'info_tambahan'
    DITAMBAH: menempelkan anotasi [Izin parsial X] pada baris detail jika tanggal tsb punya izin parsial.
    (Tidak mempengaruhi status 'bersih' — izin parsial hanya info.)
    """
    if scan_df is None or scan_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = scan_df.copy()
    for col, default in [
        ("nip",""),("nama",""),("jabatan",""),("departemen",""),
        ("jadwal_kategori",""),("is_libur",False),
        ("terlambat_menit",0),("scan_masuk",np.nan),("scan_pulang",np.nan),
        ("tanggal", pd.NaT),
    ]:
        if col not in df.columns: df[col] = default

    if not np.issubdtype(df["tanggal"].dtype, np.datetime64):
        df["tanggal"] = parse_tanggal_series(df["tanggal"])

    # --- siapkan lookup izin parsial -> (durasi) per (nip, tanggal_date)
    izin_lookup = {}
    if isinstance(izin_marks_df, pd.DataFrame) and not izin_marks_df.empty:
        tmp = izin_marks_df.copy()
        if not np.issubdtype(tmp["tanggal"].dtype, np.datetime64):
            tmp["tanggal"] = parse_tanggal_series(tmp["tanggal"])
        tmp = tmp[(tmp['tandai_ijin'] == True) & (tmp['tanggal'].notna())]
        if not tmp.empty:
            tmp['tanggal_date'] = pd.to_datetime(tmp['tanggal']).dt.date
            for _, r in tmp.iterrows():
                k = (str(r['nip']), r['tanggal_date'])
                izin_lookup[k] = str(r.get('durasi_ijin','')).strip() or "-"

    absen_cats = {"TIDAK_HADIR","IZIN_PRIBADI","SAKIT_TANPA_SKD","SAKIT_SKD"}
    kerja_mask = (~df['is_libur']) & (~df['jadwal_kategori'].isin(["LIBUR","IZIN_DINAS"]))

    nice_label = {
        "TIDAK_HADIR": "Tidak hadir",
        "IZIN_PRIBADI": "Izin pribadi",
        "SAKIT_TANPA_SKD": "Sakit tanpa SKD",
        "SAKIT_SKD": "Sakit dengan SKD",
        "IZIN_DINAS": "Izin dinas (dikecualikan)",
        "LIBUR": "Libur",
    }

    def _row_reasons(row):
        """Kembalikan daftar 'alasan' yang mempengaruhi status bersih."""
        if not kerja_mask.loc[row.name]:
            return []
        reasons = []
        kat = str(row.get('jadwal_kategori',"")).upper()
        tm = int(row.get('terlambat_menit',0) or 0)

        if kat in absen_cats:
            reasons.append(nice_label.get(kat, kat.title().replace("_"," ")))

        # Telat hanya dihitung jika bukan sakit
        if kat not in ("SAKIT_SKD","SAKIT_TANPA_SKD") and tm > toleransi_rajin_menit:
            reasons.append(f"Telat {tm}m")

        # Abaikan cek scan pada hari sakit
        if kat not in ("SAKIT_SKD","SAKIT_TANPA_SKD"):
            if _is_missing_scan(row.get('scan_masuk')):
                reasons.append("Tidak scan MASUK")
            if _is_missing_scan(row.get('scan_pulang')):
                reasons.append("Tidak scan PULANG")

        return reasons

    df['alasan_list'] = df.apply(_row_reasons, axis=1)
    df['bersih'] = kerja_mask & df['alasan_list'].apply(lambda L: len(L)==0)

    grp_keys = ['nip','nama','jabatan','departemen'] if 'departemen' in df.columns else ['nip','nama','jabatan']
    grp = df.groupby(grp_keys, dropna=False)
    jumlah_hari_kerja = grp.apply(lambda g: int(kerja_mask.loc[g.index].sum())).rename('jumlah_hari_kerja')
    jumlah_bersih = grp['bersih'].sum().astype(int).rename('jumlah_bersih')

    rekap_rajin = (pd.concat([jumlah_hari_kerja, jumlah_bersih], axis=1).reset_index())
    rekap_rajin['status'] = rekap_rajin.apply(
        lambda r: 'Rajin' if r['jumlah_hari_kerja'] == r['jumlah_bersih'] else 'Tidak', axis=1
    )

    def _agg_detail(g: pd.DataFrame) -> Tuple[str, str]:
        lines = []
        for _, row in g.iterrows():
            if kerja_mask.loc[row.name] and row['alasan_list']:
                tgl_ts = row['tanggal']
                tgl = "-" if pd.isna(tgl_ts) else pd.to_datetime(tgl_ts).strftime("%d-%m-%y")
                alasan = ", ".join(row['alasan_list'])

                # ---- Tambah anotasi jika ada izin parsial pada tanggal tsb (TIDAK mempengaruhi 'bersih') ----
                dur = ""
                if not pd.isna(tgl_ts):
                    key = (str(row.get('nip', "")), pd.to_datetime(tgl_ts).date())
                    if key in izin_lookup:
                        d = izin_lookup[key]
                        dur = f" [Izin parsial {d}]"

                lines.append(f"{tgl}: {alasan}{dur}")

        # info tambahan (libur/dinas) tetap dipisah
        info_lines = []
        sub = g[g['jadwal_kategori'].isin(["IZIN_DINAS","LIBUR"])]
        if not sub.empty:
            for _, r in sub.sort_values('tanggal').iterrows():
                tgl = "-" if pd.isna(r['tanggal']) else pd.to_datetime(r['tanggal']).strftime("%d-%m-%y")
                info_lines.append(f"{tgl}: {r['jadwal_kategori']}")

        return pd.Series({
            "alasan_tidak_rajin": "\n".join(lines) if lines else "",
            "info_tambahan": "\n".join(info_lines) if info_lines else ""
        })

    detail = grp.apply(_agg_detail).reset_index()
    detail_tidak_rajin = detail.merge(
        rekap_rajin[grp_keys + ['jumlah_hari_kerja','jumlah_bersih','status']],
        on=grp_keys, how='left'
    )
    detail_tidak_rajin = detail_tidak_rajin[detail_tidak_rajin['status'] != 'Rajin'].reset_index(drop=True)

    return rekap_rajin, detail_tidak_rajin

# =======================
# Overlay: Terapkan STATUS MARKS (Izin Full / Sakit)
# =======================
def apply_status_marks(base_df: pd.DataFrame, marks_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty or marks_df is None or marks_df.empty:
        return base_df

    df = base_df.copy()
    df['tanggal_str'] = pd.to_datetime(df['tanggal']).dt.strftime("%d-%m-%y")
    df['row_key'] = df['nip'].astype(str) + '|' + df['tanggal_str']

    mk = marks_df.copy()
    if not np.issubdtype(mk['tanggal'].dtype, np.datetime64):
        mk['tanggal'] = parse_tanggal_series(mk['tanggal'])
    mk['tanggal_str'] = pd.to_datetime(mk['tanggal']).dt.strftime("%d-%m-%y")
    mk['row_key'] = mk['nip'].astype(str) + '|' + mk['tanggal_str']
    mk = mk[mk['status'].astype(str).str.len() > 0]

    if mk.empty:
        df.drop(columns=['tanggal_str','row_key'], errors='ignore', inplace=True)
        return df

    map_status = mk.set_index('row_key')['status']
    mask = df['row_key'].isin(map_status.index)
    if mask.any():
        df.loc[mask, 'jadwal_kategori'] = df.loc[mask, 'row_key'].map(map_status)

    mask_sakit = df['jadwal_kategori'].isin(['SAKIT_SKD', 'SAKIT_TANPA_SKD'])
    if mask_sakit.any():
        df.loc[mask_sakit, ['scan_masuk', 'scan_pulang']] = np.nan
        df.loc[mask_sakit, 'terlambat_menit'] = 0

    df.drop(columns=['tanggal_str','row_key'], errors='ignore', inplace=True)
    return df

# =======================
# App
# =======================
def main():
    st.set_page_config(page_title="Aplikasi Rekap Gaji Karyawan (toleransi kejadian)", page_icon="💰", layout="wide")
    st.title("💰 Aplikasi Rekap Gaji Karyawan")
    st.caption("Tanggal dd-mm-yyyy dioptimalkan & pemrosesan hanya saat tombol ditekan (anti loading terus).")

    # -------- Sidebar
    with st.sidebar:
        st.header("📁 Upload")
        uploaded_scanlog = st.file_uploader("Upload Scanlog", type=['xlsx','xls','csv'])

        st.subheader("⏳ Toleransi Telat (Kejadian/Periode)")
        toleransi_telat_kejadian = st.number_input("Jumlah kejadian telat yang dimaafkan", min_value=0, value=3, step=1)

        st.subheader("⭐ Aturan Rajin (per-hari)")
        toleransi_rajin_menit = st.number_input("Toleransi telat per hari (menit) untuk status 'Rajin'", min_value=0, value=0, step=1)

        st.subheader("💵 Gaji Per Hari")
        default_gph = st.number_input("Default Gaji Per Hari (Rp)", min_value=0, value=0, step=1000)

        st.divider()
        if st.button("✅ Hitung Ulang Rekap (Global)", use_container_width=True):
            st.session_state.must_compute = True

    if uploaded_scanlog is None:
        st.info("Silakan upload file scanlog untuk mulai.")
        return

    scanlog_df = load_data(uploaded_scanlog)
    if scanlog_df is None:
        return

    st.success(f"Scanlog dimuat: {len(scanlog_df):,} baris")

    # ========== Filter Data ==========
    with st.expander("🎛️ Filter Data", expanded=True):
        if 'tanggal' in scanlog_df.columns and scanlog_df['tanggal'].notna().any():
            tmin_dt = pd.to_datetime(scanlog_df['tanggal']).min().date()
            tmax_dt = pd.to_datetime(scanlog_df['tanggal']).max().date()
        else:
            tmin_dt = date.today(); tmax_dt = date.today()

        mode = st.radio("Metode input tanggal", ["Kalender", "Manual (dd-mm-yy)"], horizontal=True)

        def _parse_ddmmyy(s: str) -> date:
            s = (s or "").strip()
            for fmt in ["%d-%m-%y", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y", "%d.%m.%y", "%d.%m.%Y"]:
                try:
                    return datetime.strptime(s, fmt).date()
                except Exception:
                    pass
            return pd.to_datetime(s, dayfirst=True, errors="raise").date()

        if mode == "Kalender":
            start_date, end_date = st.date_input(
                "Rentang Tanggal (DD-MM-YYYY)",
                value=(tmin_dt, tmax_dt),
                min_value=tmin_dt, max_value=tmax_dt
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                s_start = st.text_input("Tanggal mulai (dd-mm-yy)", tmin_dt.strftime("%d-%m-%y"))
            with c2:
                s_end = st.text_input("Tanggal akhir (dd-mm-yy)", tmax_dt.strftime("%d-%m-%y"))
            try:
                start_date = _parse_ddmmyy(s_start)
                end_date   = _parse_ddmmyy(s_end)
            except Exception as e:
                st.error(f"Input tanggal manual salah: {e}")
                st.stop()

        st.write(f"Periode filter (dd-mm-yy): **{start_date.strftime('%d-%m-%y')} → {end_date.strftime('%d-%m-%y')}**")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dept_opsi = sorted(scanlog_df.get('departemen', pd.Series([], dtype=str)).dropna().unique().tolist())
            f_dept = st.multiselect("Departemen", dept_opsi)
        with c2:
            jab_opsi = sorted(scanlog_df.get('jabatan', pd.Series([], dtype=str)).dropna().unique().tolist())
            f_jabatan = st.multiselect("Jabatan", jab_opsi, default=jab_opsi)
        with c3:
            nama_opsi = sorted(scanlog_df.get('nama', pd.Series([], dtype=str)).dropna().unique().tolist())
            f_nama = st.multiselect("Nama", nama_opsi)
        with c4:
            kat_opsi = ["", "LIBUR","IZIN_DINAS","TIDAK_HADIR","IZIN_PRIBADI","SAKIT_TANPA_SKD","SAKIT_SKD"]
            f_kategori = st.multiselect("Kategori Jadwal", kat_opsi)

    _df = scanlog_df
    if 'tanggal' in _df.columns:
        mask_date = _df['tanggal'].dt.date.between(start_date, end_date)
        _df = _df[mask_date]
    if f_dept:
        _df = _df[_df['departemen'].isin(f_dept)]
    if f_jabatan:
        _df = _df[_df['jabatan'].isin(f_jabatan)]
    if f_nama:
        _df = _df[_df['nama'].isin(f_nama)]
    if f_kategori:
        _df = _df[_df['jadwal_kategori'].isin(f_kategori)]

    # ---------- Preview ----------
    with st.expander("🔍 Preview Data Scanlog (setelah filter)"):
        colA, colB, colC, colD = st.columns([1,1,2,1])
        with colA: st.metric("Records", f"{len(_df):,}")
        with colB: st.metric("Karyawan unik", f"{_df['nip'].nunique() if 'nip' in _df.columns else 0:,}")
        with colC:
            st.write("**Rentang tanggal**")
            if 'tanggal' in _df.columns and _df['tanggal'].notna().any():
                _tmin = pd.to_datetime(_df['tanggal'].min())
                _tmax = pd.to_datetime(_df['tanggal'].max())
                st.code(f"{_tmin.strftime('%d-%m-%y')} → {_tmax.strftime('%d-%m-%y')}")
            else:
                st.code("-")
        with colD:
            telat_pos = int((_df.get('terlambat_menit', 0) > 0).sum())
            st.metric("Hari telat (>0m)", f"{telat_pos:,}")

        df_prev = _df.copy()
        if 'tanggal' in df_prev.columns:
            df_prev['tanggal'] = pd.to_datetime(df_prev['tanggal'], errors='coerce').dt.strftime("%d-%m-%y")
        st.dataframe(df_prev, use_container_width=True)

    # ========== Gaji Per Hari editor ==========
    karyawan_master = _df[['nip','nama','jabatan','departemen']].drop_duplicates().reset_index(drop=True)
    if 'gph_df' not in st.session_state:
        base = karyawan_master.copy(); base['gaji_per_hari'] = int(default_gph)
        st.session_state.gph_df = base
    else:
        exist = st.session_state.gph_df[['nip']].astype(str)
        add = karyawan_master[~karyawan_master['nip'].astype(str).isin(exist['nip'])].copy()
        if not add.empty:
            add['gaji_per_hari'] = int(default_gph)
            st.session_state.gph_df = pd.concat([st.session_state.gph_df, add], ignore_index=True)

    # ========== [SECTION ▶ IZIN PARSIAL] ==========
    kerja_base = (~_df['is_libur']) & (~_df['jadwal_kategori'].isin(["LIBUR","IZIN_DINAS"]))
    izin_editor_df = _df.loc[kerja_base, ['tanggal','nip','nama','jabatan','departemen','jadwal_kategori','scan_masuk','scan_pulang','terlambat_menit']].copy()
    if not izin_editor_df.empty:
        izin_editor_df['tanggal_str'] = pd.to_datetime(izin_editor_df['tanggal']).dt.strftime("%d-%m-%y")
        izin_editor_df['tanggal_fmt'] = izin_editor_df['tanggal_str']
        izin_editor_df['tandai_ijin'] = False
        izin_editor_df['durasi_ijin'] = ""
        izin_editor_df['row_key'] = izin_editor_df['nip'].astype(str) + '|' + izin_editor_df['tanggal_str']

    if 'izin_marks' not in st.session_state:
        st.session_state.izin_marks = pd.DataFrame(columns=['row_key','tanggal','nip','tandai_ijin','durasi_ijin'])

    if not izin_editor_df.empty and not st.session_state.izin_marks.empty:
        old = st.session_state.izin_marks.set_index('row_key')
        izin_editor_df = izin_editor_df.set_index('row_key')
        inter = izin_editor_df.index.intersection(old.index)
        if len(inter) > 0:
            izin_editor_df.loc[inter, 'tandai_ijin'] = old.loc[inter, 'tandai_ijin'].astype(bool)
            izin_editor_df.loc[inter, 'durasi_ijin'] = old.loc[inter, 'durasi_ijin'].astype(str)
        izin_editor_df = izin_editor_df.reset_index()

    # ========== [SECTION ▶ STATUS MARKS] ==========
    status_view_df = _df[['tanggal','nip','nama','jabatan','departemen','jadwal_kategori']].copy()
    if not status_view_df.empty:
        status_view_df['tanggal_str'] = pd.to_datetime(status_view_df['tanggal']).dt.strftime("%d-%m-%y")
        status_view_df['tanggal_fmt'] = status_view_df['tanggal_str']
        status_view_df['status'] = ""
        status_view_df['row_key'] = status_view_df['nip'].astype(str) + '|' + status_view_df['tanggal_str']

    if 'status_marks' not in st.session_state:
        st.session_state.status_marks = pd.DataFrame(columns=['row_key','tanggal','nip','status'])

    if not status_view_df.empty and not st.session_state.status_marks.empty:
        old = st.session_state.status_marks.set_index('row_key')
        status_view_df = status_view_df.set_index('row_key')
        inter = status_view_df.index.intersection(old.index)
        if len(inter) > 0:
            status_view_df.loc[inter, 'status'] = old.loc[inter, 'status'].astype(str)
        status_view_df = status_view_df.reset_index()

    # ========== Tabs ==========
    tab_status, tab_izin, tab_gph, tab_rekap, tab_output, tab_rajin = st.tabs(
        ["🩺 Status Izin/Sakit (Full)", "📝 Izin Parsial", "💵 Gaji Per Hari", "📊 Rekap & Statistik", "✅ Output Akhir", "⭐ Karyawan Rajin"]
    )

    with tab_status:
        st.markdown("### Tandai Status per Tanggal: **Izin (Full)**, **Sakit SKD**, **Sakit non-SKD**")
        st.caption("Tanda ini **menimpa** kolom `jadwal_kategori`. Hari sakit otomatis mengabaikan scan & telat.")
        if status_view_df.empty:
            st.info("Tidak ada baris pada filter saat ini.")
        else:
            cols_show = ['row_key','tanggal_fmt','nip','nama','jabatan','departemen','jadwal_kategori','status']
            edited_status = st.data_editor(
                status_view_df[cols_show],
                use_container_width=True,
                column_config={
                    "tanggal_fmt": st.column_config.TextColumn("Tanggal (dd-mm-yy)"),
                    "status": st.column_config.SelectboxColumn("Tandai Status", options=["", "IZIN_PRIBADI", "SAKIT_SKD", "SAKIT_TANPA_SKD"]),
                },
                hide_index=True,
                key="editor_status_marks"
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Simpan Tanda (tanpa hitung)", key="btn_save_status"):
                    base_map = status_view_df[['row_key','tanggal']].drop_duplicates()
                    st.session_state.status_marks = edited_status[['row_key','nip','status']].merge(base_map, on='row_key', how='left')
                    st.success("Tanda disimpan. Tekan 'Hitung Ulang Rekap' jika ingin memproses.")
            with c2:
                if st.button("✅ Simpan & Hitung Ulang Rekap", key="btn_apply_status"):
                    base_map = status_view_df[['row_key','tanggal']].drop_duplicates()
                    st.session_state.status_marks = edited_status[['row_key','nip','status']].merge(base_map, on='row_key', how='left')
                    st.session_state.must_compute = True

    with tab_izin:
        st.markdown("### Tandai Izin **Parsial** per Hari")
        st.caption("Centang **Tandai Izin**, pilih durasi (0–5 jam). Tidak mempengaruhi `jadwal_kategori`.")
        if izin_editor_df.empty:
            st.info("Tidak ada baris hari kerja pada filter saat ini.")
        else:
            shown_cols = ['row_key','tanggal_fmt','nip','nama','jabatan','departemen','scan_masuk','scan_pulang','terlambat_menit','tandai_ijin','durasi_ijin']
            edited_parsial = st.data_editor(
                izin_editor_df[shown_cols],
                use_container_width=True,
                column_config={
                    "tanggal_fmt": st.column_config.TextColumn("Tanggal (dd-mm-yy)"),
                    "tandai_ijin": st.column_config.CheckboxColumn("Tandai Izin"),
                    "durasi_ijin": st.column_config.SelectboxColumn("Durasi Izin", options=["", "0-2 jam","2-3 jam","3-4 jam","4-5 jam"]),
                },
                hide_index=True,
                key="editor_izin_parsial"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Simpan Izin Parsial (tanpa hitung)", key="btn_save_parsial"):
                    base_map = izin_editor_df[['row_key','tanggal']].drop_duplicates()
                    st.session_state.izin_marks = edited_parsial[['row_key','nip','tandai_ijin','durasi_ijin']].merge(base_map, on='row_key', how='left')
                    st.success("Izin parsial disimpan. Tekan 'Hitung Ulang Rekap' jika ingin memproses.")
            with c2:
                if st.button("✅ Simpan & Hitung Ulang Rekap", key="btn_apply_parsial"):
                    base_map = izin_editor_df[['row_key','tanggal']].drop_duplicates()
                    st.session_state.izin_marks = edited_parsial[['row_key','nip','tandai_ijin','durasi_ijin']].merge(base_map, on='row_key', how='left')
                    st.session_state.must_compute = True

    with tab_gph:
        st.markdown("### Editor Gaji Per Hari per Karyawan")
        st.caption("View ini otomatis mengikuti filter utama di atas (periode, departemen, jabatan, nama, kategori).")
    
        # --- Bangun master karyawan TERFILTER dari _df (sudah terfilter dengan filter utama)
        _tmp = _df.copy()
        for col in ['nip','nama','jabatan','departemen']:
            if col not in _tmp.columns:
                _tmp[col] = ""  # safety jika kolom tidak ada
    
        master_view = _tmp[['nip','nama','jabatan','departemen']].drop_duplicates().copy()
        master_view['nip'] = master_view['nip'].astype(str)
    
        # --- Sinkronkan daftar karyawan ke session_state.gph_df (global)
        if 'gph_df' not in st.session_state or st.session_state.gph_df is None or st.session_state.gph_df.empty:
            base = master_view.copy()
            base['gaji_per_hari'] = int(default_gph)
            st.session_state.gph_df = base
        else:
            # Pastikan kolom wajib ada
            for col in ['nip','nama','jabatan','departemen','gaji_per_hari']:
                if col not in st.session_state.gph_df.columns:
                    st.session_state.gph_df[col] = "" if col != 'gaji_per_hari' else int(default_gph)
            st.session_state.gph_df['nip'] = st.session_state.gph_df['nip'].astype(str)
    
            # Tambahkan karyawan baru yang belum ada di gph_df
            exist_nips = set(st.session_state.gph_df['nip'])
            add_rows = master_view[~master_view['nip'].isin(exist_nips)].copy()
            if not add_rows.empty:
                add_rows['gaji_per_hari'] = int(default_gph)
                st.session_state.gph_df = pd.concat([st.session_state.gph_df, add_rows], ignore_index=True)
    
        # --- Buat VIEW sesuai filter utama
        full_gph = st.session_state.gph_df.copy()
        full_gph['nip'] = full_gph['nip'].astype(str)
    
        view = full_gph.merge(
            master_view[['nip','nama','jabatan','departemen']],
            on='nip', how='inner', suffixes=('', '_m')
        )
    
        # Rapikan kolom setelah merge
        for col in ['nama','jabatan','departemen']:
            if f"{col}_m" in view.columns:
                view[col] = np.where(view[col].astype(str).str.strip().eq(""), view[f"{col}_m"], view[col])
                view.drop(columns=[f"{col}_m"], inplace=True, errors='ignore')
    
        # Safety: pastikan kolom ada
        for col in ['departemen','jabatan','nama']:
            if col not in view.columns:
                view[col] = ""
    
        # --- Urutkan rapi
        sort_cols = [c for c in ['departemen','jabatan','nama','nip'] if c in view.columns]
        view = view[[c for c in ['nip','nama','jabatan','departemen','gaji_per_hari'] if c in view.columns]].sort_values(sort_cols)
    
        st.write(f"Baris tampil: **{len(view):,}** (mengikuti filter utama)")
        gph_edit_view = st.data_editor(
            view,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "gaji_per_hari": st.column_config.NumberColumn("Gaji Per Hari (Rp)", min_value=0, step=1000)
            },
            hide_index=True,
            key="gph_editor_view"
        )
    
        # --- Update partial: hanya apply perubahan pada NIP yang tampil
        def _apply_update_to_state(edited_df: pd.DataFrame):
            if edited_df is None or edited_df.empty or 'nip' not in edited_df.columns:
                return
            edited_df = edited_df.copy()
            edited_df['nip'] = edited_df['nip'].astype(str)
            if 'gaji_per_hari' not in edited_df.columns:
                return
            upd_map = edited_df.set_index('nip')['gaji_per_hari'].to_dict()
    
            base = st.session_state.gph_df.copy()
            base['nip'] = base['nip'].astype(str)
            mask = base['nip'].isin(upd_map.keys())
            if mask.any():
                base.loc[mask, 'gaji_per_hari'] = base.loc[mask, 'nip'].map(upd_map)
            st.session_state.gph_df = base
    
        c1b, c2b = st.columns(2)
        with c1b:
            if st.button("💾 Simpan GPH (tanpa hitung)", key="btn_save_gph_filtered"):
                _apply_update_to_state(gph_edit_view)
                st.success("Gaji per hari disimpan (subset sesuai filter).")
        with c2b:
            if st.button("✅ Simpan & Hitung Ulang Rekap", key="btn_apply_gph_filtered"):
                _apply_update_to_state(gph_edit_view)
                st.session_state.must_compute = True
    
        st.divider()
        # Download GPH (global + view filter)
        buf_gph = io.BytesIO()
        with pd.ExcelWriter(buf_gph, engine='openpyxl') as writer:
            cols_global = [c for c in ['nip','nama','jabatan','departemen','gaji_per_hari'] if c in st.session_state.gph_df.columns]
            st.session_state.gph_df[cols_global].sort_values([c for c in ['departemen','jabatan','nama','nip'] if c in cols_global]).to_excel(
                writer, sheet_name='Gaji_Per_Hari_Global', index=False
            )
            cols_view = [c for c in ['nip','nama','jabatan','departemen','gaji_per_hari'] if c in gph_edit_view.columns]
            gph_edit_view[cols_view].to_excel(writer, sheet_name='Gaji_Per_Hari_View_Filter', index=False)
        buf_gph.seek(0)
        st.download_button(
            "⬇️ Download GPH (Global & View Filter)",
            data=buf_gph,
            file_name=f"gaji_per_hari_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )



    gph_map = {}
    if not st.session_state.gph_df.empty:
        gph_map = {str(r['nip']): float(r['gaji_per_hari'] or 0) for _, r in st.session_state.gph_df.iterrows()}

    status_marks_df = st.session_state.status_marks.copy() if isinstance(st.session_state.status_marks, pd.DataFrame) and not st.session_state.status_marks.empty else pd.DataFrame()
    izin_marks_df = st.session_state.izin_marks.copy() if isinstance(st.session_state.izin_marks, pd.DataFrame) and not st.session_state.izin_marks.empty else pd.DataFrame()

    _df_effective = apply_status_marks(_df, status_marks_df)

    filter_key = "|".join([
        f"{start_date}~{end_date}",
        ",".join(sorted([str(x) for x in (f_dept or [])])),
        ",".join(sorted([str(x) for x in (f_jabatan or [])])),
        ",".join(sorted([str(x) for x in (f_nama or [])])),
        ",".join(sorted([str(x) for x in (f_kategori or [])])),
        str(len(_df_effective))
    ])
    if st.session_state.get("last_filter_key") != filter_key:
        st.session_state.last_filter_key = filter_key
        st.session_state.must_compute = True

    if 'rekap_df_cache' not in st.session_state or st.session_state.get('must_compute', False):
        with st.status("Menghitung rekap & rajin...", state="running"):
            rekap_df_calc = process_payroll_data(
                _df_effective, gaji_per_hari_map=gph_map,
                toleransi_telat_kejadian=int(toleransi_telat_kejadian),
                izin_pribadi_marks=izin_marks_df
            )
            # >>> PERUBAHAN: kirim izin_marks_df ke build_rajin_recap agar muncul anotasi Izin Parsial
            rekap_rajin_calc, detail_tidak_rajin_calc = build_rajin_recap(
                _df_effective, toleransi_rajin_menit=toleransi_rajin_menit, izin_marks_df=izin_marks_df
            )
        st.session_state.rekap_df_cache = rekap_df_calc
        st.session_state.rekap_rajin_cache = rekap_rajin_calc
        st.session_state.detail_tidak_rajin_cache = detail_tidak_rajin_calc
        st.session_state.must_compute = False

    rekap_df = st.session_state.get('rekap_df_cache', pd.DataFrame())
    rekap_rajin_cache = st.session_state.get('rekap_rajin_cache', pd.DataFrame())
    detail_tidak_rajin_cache = st.session_state.get('detail_tidak_rajin_cache', pd.DataFrame())

    with tab_rekap:
        st.markdown("## 📊 Hasil Rekap (Periode Terfilter — 1 baris/karyawan)")
        if rekap_df is None or rekap_df.empty:
            st.info("Belum ada data rekap untuk ditampilkan.")
        else:
            r1, r2, r3 = st.columns(3)
            with r1:
                f_dept_rekap = st.multiselect("Filter Departemen", sorted(rekap_df['departemen'].dropna().unique().tolist()))
            with r2:
                f_jabatan_rekap = st.multiselect("Filter Jabatan", sorted(rekap_df['jabatan'].dropna().unique().tolist()))
            with r3:
                f_nama_rekap = st.multiselect("Filter Nama", sorted(rekap_df['nama'].dropna().unique().tolist()))

            _rd = rekap_df.copy()
            if f_dept_rekap:
                _rd = _rd[_rd['departemen'].isin(f_dept_rekap)]
            if f_jabatan_rekap:
                _rd = _rd[_rd['jabatan'].isin(f_jabatan_rekap)]
            if f_nama_rekap:
                _rd = _rd[_rd['nama'].isin(f_nama_rekap)]

            st.markdown("### 💰 Rekap Gaji & Potongan")
            display_columns = [
                'nip','nama','jabatan','departemen','periode_tanggal',
                'gaji_per_hari','hari_hadir','gaji_bruto',
                'toleransi_telat_awal','toleransi_telat_terpakai_miss_in','toleransi_telat_sisa',
                'total_telat_kejadian','telat_dipotong',
                'potongan_terlambat','potongan_absen','potongan_izin_pribadi_parsial',
                'total_potongan','gaji_akhir','alasan_potongan'
            ]
            display_columns = [c for c in display_columns if c in _rd.columns]
            st.dataframe(_rd[display_columns], use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_karyawan = int(_rd['nip'].nunique())
                st.metric("Total Karyawan", total_karyawan)
            with col2:
                total_potongan = int(_rd['total_potongan'].sum()) if 'total_potongan' in _rd.columns else 0
                st.metric("Total Potongan", f"Rp {total_potongan:,}")
            with col3:
                total_bruto = int(_rd['gaji_bruto'].sum()) if 'gaji_bruto' in _rd.columns else 0
                st.metric("Total Gaji Bruto (hadir)", f"Rp {total_bruto:,}")
            with col4:
                total_akhir = int(_rd['gaji_akhir'].sum()) if 'gaji_akhir' in _rd.columns else 0
                st.metric("Total Gaji Akhir", f"Rp {total_akhir:,}")

            st.divider()
            st.subheader("⬇️ Download")
            rekap_buf = io.BytesIO()
            with pd.ExcelWriter(rekap_buf, engine='openpyxl') as writer:
                _rd[display_columns].to_excel(writer, sheet_name='Rekap_Per_Karyawan', index=False)
            rekap_buf.seek(0)
            st.download_button(
                "📥 Download REKAP (Excel)",
                data=rekap_buf,
                file_name=f"rekap_per_karyawan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            stat_buf = io.BytesIO()
            stat_global_rows = []
            stat_global_rows.append(["Total Karyawan", total_karyawan])
            stat_global_rows.append(["Total Potongan", total_potongan])
            stat_global_rows.append(["Total Gaji Bruto (hadir)", total_bruto])
            stat_global_rows.append(["Total Gaji Akhir", total_akhir])
            df_stat_global = pd.DataFrame(stat_global_rows, columns=["Metric", "Value"])

            kolom_sum = [c for c in [
                'hari_hadir','gaji_bruto',
                'potongan_terlambat','potongan_absen','potongan_izin_pribadi_parsial',
                'total_potongan','gaji_akhir'
            ] if c in _rd.columns]

            if 'departemen' in _rd.columns and not _rd.empty:
                try:
                    df_per_dept = _rd.groupby('departemen', dropna=False).agg(
                        **({k: (k, 'sum') for k in kolom_sum}),
                        jumlah_karyawan=('nip', 'nunique')
                    ).reset_index()
                except Exception:
                    df_per_dept = _rd.groupby('departemen', dropna=False).agg(
                        {**{k: 'sum' for k in kolom_sum}, 'nip': 'nunique'}
                    ).reset_index().rename(columns={'nip':'jumlah_karyawan'})
            else:
                df_per_dept = pd.DataFrame()

            if 'jabatan' in _rd.columns and not _rd.empty:
                try:
                    df_per_jabatan = _rd.groupby('jabatan', dropna=False).agg(
                        **({k: (k, 'sum') for k in kolom_sum}),
                        jumlah_karyawan=('nip', 'nunique')
                    ).reset_index()
                except Exception:
                    df_per_jabatan = _rd.groupby('jabatan', dropna=False).agg(
                        {**{k: 'sum' for k in kolom_sum}, 'nip': 'nunique'}
                    ).reset_index().rename(columns={'nip':'jumlah_karyawan'})
            else:
                df_per_jabatan = pd.DataFrame()

            telat_cols = [c for c in ['total_telat_kejadian','telat_dipotong'] if c in _rd.columns]
            if telat_cols:
                df_telat = _rd[['nip','nama'] + telat_cols].copy()
            else:
                df_telat = pd.DataFrame()

            with pd.ExcelWriter(stat_buf, engine='openpyxl') as writer:
                df_stat_global.to_excel(writer, sheet_name='Statistik_Global', index=False)
                if not df_per_dept.empty:
                    df_per_dept.to_excel(writer, sheet_name='Per_Departemen', index=False)
                if not df_per_jabatan.empty:
                    df_per_jabatan.to_excel(writer, sheet_name='Per_Jabatan', index=False)
                if not df_telat.empty:
                    df_telat.to_excel(writer, sheet_name='Telat_Per_Karyawan', index=False)

            stat_buf.seek(0)
            st.download_button(
                "📊 Download STATISTIK (Excel)",
                data=stat_buf,
                file_name=f"statistik_rekap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with tab_output:
        st.markdown("## ✅ Output Akhir (Pendapatan per Karyawan — periode terfilter)")
        if rekap_df is None or rekap_df.empty:
            st.info("Belum ada data rekap.")
        else:
            ak = rekap_df
            kolom_output = [
                'nip','nama','jabatan','departemen',
                'gaji_per_hari','hari_hadir','gaji_bruto',
                'potongan_terlambat','potongan_absen','potongan_izin_pribadi_parsial',
                'total_potongan','gaji_akhir'
            ]
            kolom_output = [c for c in kolom_output if c in ak.columns]
            st.dataframe(ak[kolom_output], use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Karyawan", ak['nip'].nunique())
            with c2:
                st.metric("Total Jumlah Pendapatan (Gaji Akhir)", f"Rp {int(ak['gaji_akhir'].sum()):,}")

            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine='openpyxl') as writer:
                ak[kolom_output].to_excel(writer, sheet_name='Output_Akhir', index=False)
            out_buf.seek(0)
            st.download_button("📥 Download Output Akhir (Excel)",
                               out_buf,
                               file_name=f"output_akhir_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab_rajin:
        st.markdown("## ⭐ Rekapan Karyawan Rajin (berdasarkan Scanlog aktif)")
        st.caption("Definisi rajin: semua HARI KERJA bersih (tidak telat > ambang menit, tidak absen/izin/sakit, dan scan masuk/pulang lengkap).")

        rekap_rajin = rekap_rajin_cache
        detail_tidak_rajin = detail_tidak_rajin_cache

        if rekap_rajin is None or rekap_rajin.empty:
            st.info("Belum ada data untuk dihitung.")
        else:
            colR1, colR2 = st.columns([2,1])
            with colR1:
                show_only_rajin = st.checkbox("Tampilkan hanya karyawan Rajin", value=True)
            with colR2:
                total_rajin = int((rekap_rajin['status'] == 'Rajin').sum())
                st.metric("Total Rajin", f"{total_rajin} / {rekap_rajin['nip'].nunique()}")

            _r = rekap_rajin
            if show_only_rajin:
                _r = _r[_r['status'] == 'Rajin']

            cols_rekap = ['nip','nama','jabatan']
            if 'departemen' in _r.columns:
                cols_rekap.append('departemen')
            cols_rekap += ['jumlah_hari_kerja','jumlah_bersih','status']
            st.markdown("### 📋 Rekap Rajin")
            st.dataframe(_r[cols_rekap], use_container_width=True)

            buf_r = io.BytesIO()
            with pd.ExcelWriter(buf_r, engine='openpyxl') as writer:
                _r[cols_rekap].to_excel(writer, sheet_name='Rekap_Rajin', index=False)
            buf_r.seek(0)
            st.download_button("⬇️ Download Rekap Rajin (Excel)",
                               buf_r,
                               file_name=f"rekap_karyawan_rajin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("### 🧾 Detail Alasan Karyawan Tidak Rajin")
            if detail_tidak_rajin is None or detail_tidak_rajin.empty:
                st.success("Semua karyawan berstatus **Rajin** pada rentang & filter saat ini 🎉")
            else:
                cols_det = ['nip','nama','jabatan']
                if 'departemen' in detail_tidak_rajin.columns:
                    cols_det.append('departemen')
                cols_det += ['jumlah_hari_kerja','jumlah_bersih','status','alasan_tidak_rajin','info_tambahan']
                cols_det = [c for c in cols_det if c in detail_tidak_rajin.columns]

                st.dataframe(detail_tidak_rajin[cols_det], use_container_width=True)

                buf_d = io.BytesIO()
                with pd.ExcelWriter(buf_d, engine='openpyxl') as writer:
                    detail_tidak_rajin[cols_det].to_excel(writer, sheet_name='Detail_Tidak_Rajin', index=False)
                buf_d.seek(0)
                st.download_button("⬇️ Download Detail Tidak Rajin (Excel)",
                                   buf_d,
                                   file_name=f"detail_tidak_rajin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()

