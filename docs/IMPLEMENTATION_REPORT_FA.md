# گزارش نهایی پیاده‌سازی پروژه

## تعریف دقیق مسئله

پروژه به‌صورت یک سامانه محلی و قابل‌تکرار برای **برچسب‌گذاری چندبرچسبی صداهای محیطی در سطح پنجره** پیاده‌سازی و آزمایش شده است.

تعریف علمی صحیح:

```text
Window-Level Multi-Label Audio Tagging / Event-Presence Detection
```

سامانه وجود کلاس‌های صوتی آموزش‌دیده را در هر پنجره کوتاه پیش‌بینی می‌کند و زمان دقیق شروع/پایان رویداد را تخمین نمی‌زند؛ بنابراین ادعای Full Sound Event Detection ندارد.

## پروتکل نهایی داده و آزمایش

داده اصلی:

```text
UrbanSound8K
```

تقسیم ثابت:

```text
Train: folds 1–7
Validation: fold 8
Test: folds 9–10
```

هشت کلاس هدف:

```text
air_conditioner
children_playing
dog_bark
drilling
engine_idling
jackhammer
siren
car_horn
```

دو کلاس held-out برای بررسی محدود rejection:

```text
gun_shot
street_music
```

برای جلوگیری از leakage، گروه‌های منبعی که بین splitها عبور می‌کردند از manifestهای پژوهشی حذف شدند. مخلوط‌های چندبرچسبی فقط **بعد از split** و فقط از منابع همان split ساخته شدند.

شرایط مخلوط کنترل‌شده:

```text
relative dB: -6, 0, +6
overlap: 0.25, 0.50, 1.00
```

## پیش‌پردازش و مدل‌ها

تنظیمات اصلی:

```text
sample rate = 22050 Hz
window = 2.0 s
stream hop = 1.0 s
n_fft = 1024
feature hop = 512
n_mels = 64
```

دو خانواده مدل:

```text
CNN baseline
CRNN = CNN frontend + unidirectional GRU + temporal mean pooling
```

آموزش با `BCEWithLogitsLoss` انجام می‌شود. خروجی آموزش logits خام است و sigmoid فقط برای امتیازدهی در inference/evaluation استفاده می‌شود.

## انتخاب مدل و threshold

- انتخاب checkpoint فقط با validation mAP انجام شد.
- threshold هر کلاس فقط روی validation و با معیار F1 همان کلاس انتخاب شد.
- test قبل از freeze نهایی برای انتخاب threshold یا مدل استفاده نشد.
- پس از مشاهده نتیجه held-out، هیچ retuning انجام نشد.

## آزمایش رسمی چند-seed

ماتریس رسمی:

```text
CNN  × seeds 13, 23, 37
CRNN × seeds 13, 23, 37
```

نتیجه نهایی held-out:

| معیار | CNN | CRNN |
|---|---:|---:|
| mAP | 0.618628 ± 0.008281 | **0.727378 ± 0.007039** |
| F1 micro | 0.549715 ± 0.004061 | **0.592859 ± 0.010359** |
| F1 macro | 0.564576 ± 0.003953 | **0.617833 ± 0.013461** |
| Hamming loss | 0.194428 ± 0.000791 | **0.173433 ± 0.008472** |

در پروتکل freezeشده، CRNN خانواده مدل قوی‌تر است. این نتیجه متعلق به همان پروتکل نهایی است و نباید با تغییرات پس از test «بهبود» داده شود.

## محدودیت مهم OOD

قاعده:

```text
اگر هیچ کلاس شناخته‌شده‌ای از threshold خود عبور نکند:
No confident known class
```

برای CRNN نرخ rejection کلاس‌های held-out فقط حدود:

```text
4.69%
```

و false acceptance حدود:

```text
95.31%
```

بود. بنابراین این روش یک heuristic ساده است و **Open-Set Recognition قابل‌اعتماد محسوب نمی‌شود**.

## مدل deployment

مدل deployment فقط بر اساس validation انتخاب شد:

```text
model = CRNN
seed = 23
validation mAP = 0.6757137110
```

هیچ seed با نگاه به test انتخاب نشد.

## نتیجه runtime

روی سیستم توسعه تأییدشده:

```text
CPU canonical p95  = 4.358 ms
CUDA canonical p95 = 1.561 ms
stream hop          = 1000 ms
```

هر دو مسیر شرط مهندسی عدم تشکیل backlog را پاس کردند.

تأخیر اولیه فقط چند میلی‌ثانیه نیست؛ برای اولین prediction ابتدا باید پنجره 2 ثانیه‌ای جمع‌آوری شود.

## رابط فایل و میکروفن

موارد زیر واقعاً verify شدند:

```text
تحلیل چندپنجره‌ای فایل
Gradio UI
میکروفن مرورگر
history کامل پنجره‌های live
مرتب‌سازی newest/oldest
Stop با حفظ نتایج
Clear صریح
sounddevice microphone CLI
physical microphone capture
CPU live inference
```

در صدای محیط ساکت/فن لپ‌تاپ false positive مشاهده شد. این موضوع به‌عنوان limitation ثبت شد و برای پنهان‌کردن آن thresholdها تغییر داده نشدند.

## QA، CI و clean install

Phase 14:

```text
repository QA gate = PASS
```

Phase 15:

```text
GitHub Actions CI = GREEN
```

Phase 16A در clone و `.venv` کاملاً تازه:

```text
pip check = PASS
clean-install verifier = PASS
full pytest = 173 passed
working tree = clean
```

Phase 16B:

```text
documentation contract = 8 passed
full pytest = 181 passed
final documentation commit = 539418a
GitHub Actions = GREEN
```

## مواردی که عمداً خارج از دامنه باقی ماندند

- تخمین onset/offset دقیق،
- source separation،
- source counting،
- localization،
- open-set/OOD پیشرفته،
- calibration مطالعه‌شده،
- MFCC/ZCR/RMS به‌عنوان آزمایش اصلی،
- Quantization/ONNX،
- Backend/REST API،
- Database،
- Authentication،
- Microservices/Kubernetes.

همچنین مجموعه مستقل و برچسب‌خورده real-world multi-label برای ادعای کمی درباره دقت میکروفن زنده وجود ندارد؛ بنابراین چنین ادعایی ساخته نمی‌شود.

## اجرای سریع نرم‌افزار بدون UrbanSound8K

```powershell
python scripts\run_demo_pipeline.py
```

UI با artifactهای synthetic:

```powershell
python scripts\run_ui.py `
  --checkpoint artifacts\demo_crnn\best_model.pt `
  --thresholds artifacts\demo_crnn\thresholds.json `
  --device cpu
```

UI با artifact پژوهشی freezeشده، در صورت موجودبودن محلی:

```powershell
python scripts\run_ui.py `
  --experiment-root artifacts\experiments_phase11 `
  --device cpu
```

## مرز علمی نهایی

نتیجه held-out freeze شده است.

نباید بر اساس نتیجه test یا رفتار live:

```text
threshold عوض شود
مدل/seed دوباره انتخاب شود
preprocessing تغییر کند
کلاس‌ها یا split تغییر کنند
تعریف metric تغییر کند
```

هر توسعه پژوهشی جدید باید یک protocol جدید و ازپیش‌اعلام‌شده داشته باشد.
