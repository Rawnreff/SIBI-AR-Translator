@echo off
echo =======================================================
echo Menjalankan SIBI AR Translator Local Server...
echo =======================================================
echo.
echo Pastikan Anda tidak menutup jendela ini selama menggunakan aplikasi.
echo Buka browser dan akses alamat berikut:
echo http://localhost:8080/sibi_ar.html
echo.
echo Server sedang berjalan... (Tekan Ctrl+C untuk berhenti)
python -m http.server 8080
