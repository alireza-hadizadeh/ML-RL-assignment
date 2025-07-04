
### گزارش  برنامه PCA (تحلیل مؤلفه اصلی)
#### • هدف برنامه:
کاهش ابعاد داده‌های مجموعه‌ی Iris به‌طوری که بیشترین واریانس (پراکندگی) داده‌ها حفظ شود. این روش بدون استفاده از برچسب‌های کلاس انجام می‌شود.

#### • معرفی داده‌ها:
مجموعه‌ی داده‌ی Iris شامل 150 نمونه گل در 3 کلاس (ستوسا، ورسیکالر، ویرجینیکا) است. هر نمونه دارای 4 ویژگی عددی می‌باشد:

طول کاسبرگ (Sepal Length)

عرض کاسبرگ (Sepal Width)

طول گلبرگ (Petal Length)

عرض گلبرگ (Petal Width)

#### • مراحل اجرای الگوریتم PCA:
مرکزسازی داده‌ها: از هر ستون میانگین گرفته و آن را از تمام مقادیر همان ستون کم می‌کنیم تا میانگین هر ویژگی صفر شود.

محاسبه ماتریس کوواریانس: با استفاده از داده‌های مرکز یافته، ماتریس کوواریانس 4×4 ساخته می‌شود که روابط بین ویژگی‌ها را نشان می‌دهد.

محاسبه بردارهای ویژه و مقدارهای ویژه: با حل معادله مشخصه، بردارهای ویژه (جهت‌های اصلی داده‌ها) و مقدارهای ویژه (میزان واریانس در آن جهت‌ها) به‌دست می‌آیند.

مرتب‌سازی بر اساس مقدارهای ویژه: جهت‌هایی که بیشترین واریانس را حفظ می‌کنند انتخاب می‌شوند.

پروژکت کردن داده‌ها بر روی مؤلفه‌های اصلی: با استفاده از بردارهای ویژه‌ی منتخب، داده‌ها به فضای جدید (مثلاً دوبعدی) منتقل می‌شوند.

نمایش داده‌های کاهش‌یافته روی نمودار پراکندگی (scatter).

#### • خروجی برنامه:
نموداری که داده‌ها را در فضای دوبعدی نمایش می‌دهد، به‌طوری که داده‌های مشابه (گل‌های هم‌کلاس) در نزدیکی هم و جدا از کلاس‌های دیگر قرار گرفته‌اند.

#### • مزایا:
حذف ویژگی‌های زائد و کاهش ابعاد

سرعت بیشتر برای پردازش

مناسب برای داده‌های بدون برچسب

