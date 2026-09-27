# گزارش پیاده‌سازی پروژه

## چه چیزی پیاده‌سازی شد؟

نسخه اجرایی پروژه به‌صورت یک سامانه محلی و قابل‌تکرار برای **برچسب‌گذاری چندبرچسبی صداهای محیطی در سطح پنجره** پیاده‌سازی شده است. زنجیره کامل شامل آماده‌سازی داده، تولید مخلوط‌های کنترل‌شده، پیش‌پردازش صوت، استخراج Log-Mel، مدل CNN پایه، مدل CRNN مبتنی بر GRU، آموزش، تنظیم آستانه‌ها، ارزیابی، inference روی فایل، پردازش پنجره‌ای جریانی، benchmark زمان اجرا و رابط نمایشی Gradio است.

## تغییر مهم نسبت به متن اولیه پروپوزال

نام علمی مسئله در پیاده‌سازی به‌صورت دقیق‌تر `Window-Level Multi-Label Audio Tagging / Event Presence Detection` تفسیر شده است. سامانه زمان دقیق آغاز و پایان رویداد را تخمین نمی‌زند و بنابراین ادعای Full Sound Event Detection ندارد.

همچنین خروجی قبلی «ناشناخته» به عبارت دقیق‌تر `no_confident_known_class` تبدیل شده است. این خروجی تنها می‌گوید هیچ کلاس شناخته‌شده‌ای از آستانه عبور نکرده و به‌معنای حل کامل Open-Set Recognition نیست.

## تصمیم‌های اصلی فنی

- داده اصلی: UrbanSound8K، بدون بازتوزیع داخل مخزن.
- تقسیم پیش‌فرض: Foldهای 1 تا 7 آموزش، 8 اعتبارسنجی، 9 و 10 آزمون.
- کلاس‌های اصلی: 8 کلاس قابل تنظیم.
- کلاس‌های held-out: دو کلاس برای بررسی محدود رفتار رد.
- ویژگی اصلی: Log-Mel Spectrogram.
- خط پایه: CNN سبک.
- مدل اصلی: CNN + GRU یک‌جهته + Temporal Mean Pooling.
- Loss: `BCEWithLogitsLoss`.
- Threshold: مستقل برای هر کلاس و فقط با Validation.
- معیارها: micro/macro Precision/Recall/F1، mAP، Hamming Loss و معیارهای per-class.
- آزمایش کنترل‌شده: relative level و temporal overlap.
- معیار بالدرنگ مهندسی: `p95 compute time < hop duration`.

## توصیه‌هایی که عمداً پیاده‌سازی نشدند

موارد زیر برای MVP حذف شدند تا پروژه بیش‌ازحد پیچیده نشود:

- Database؛ چون داده‌ها transactional نیستند و CSV/JSON کافی است.
- REST API و Backend مجزا؛ چون سامانه محلی است و ارزش پژوهشی اضافه نمی‌کند.
- Authentication؛ چون رابط فقط localhost است.
- Kubernetes/Microservices/Message Broker؛ کاملاً خارج از نیاز پروژه.
- MFCC/ZCR/RMS به‌عنوان مسیر اصلی؛ برای جلوگیری از شاخه‌های آزمایشی غیرضروری.
- Quantization/ONNX؛ صرفاً Future Work.
- روش‌های پیشرفته Open-Set؛ خارج از دامنه کارشناسی.

## وضعیت Verification

در محیط ساخت مخزن موارد زیر واقعاً اجرا و بررسی شدند:

- نصب editable پروژه در حالت offline با dependencyهای از قبل موجود.
- اجرای کامل تست‌ها.
- تولید dataset مصنوعی امن برای smoke test.
- آموزش واقعی CNN روی demo data.
- آموزش واقعی CRNN روی demo data.
- ارزیابی test برای هر دو مدل demo.
- inference از checkpoint ذخیره‌شده.
- runtime benchmark روی CPU.
- ساخت موفق رابط Gradio.

اعداد demo فقط برای اثبات کارکرد pipeline هستند و **نباید به‌عنوان نتیجه پژوهشی UrbanSound8K گزارش شوند**.

## چه چیزی هنوز به داده واقعی نیاز دارد؟

برای تکمیل نتیجه نهایی دانشگاهی باید UrbanSound8K تهیه و مراحل زیر اجرا شوند:

1. ساخت manifestهای رسمی.
2. اجرای CNN و CRNN برای حداقل سه seed.
3. ارزیابی frozen test.
4. محاسبه mean ± std.
5. ثبت مشخصات سیستم مرجع برای benchmark.
6. در صورت امکان، تهیه یک مجموعه کوچک واقعی و مستقل برای بررسی domain gap.

## اجرای سریع

برای تست نرم‌افزار بدون UrbanSound8K:

```bash
python scripts/run_demo_pipeline.py
```

برای رابط نمایشی:

```bash
python scripts/run_ui.py \
  --checkpoint artifacts/demo_crnn/best_model.pt \
  --thresholds artifacts/demo_crnn/thresholds.json
```

برای اجرای واقعی با UrbanSound8K ابتدا:

```bash
python scripts/prepare_urbansound8k.py \
  --dataset-root /path/to/UrbanSound8K \
  --output-dir artifacts/manifests
```

و سپس `scripts/run_experiments.py` اجرا می‌شود.
