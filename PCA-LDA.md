

##  گزارش جامع مقایسه PCA و LDA با ارزیابی طبقه‌بند

###  هدف کلی برنامه:

هدف این پروژه، **مقایسه دو روش کاهش ابعاد PCA و LDA** با استفاده از داده‌های معروف **Iris** و بررسی اثر این کاهش ابعاد در عملکرد یک مدل طبقه‌بندی (در اینجا **رگرسیون لجستیک**) می‌باشد. در پایان، از معیارهایی نظیر **دقت (Accuracy)** و **ماتریس درهم‌ریختگی (Confusion Matrix)** برای ارزیابی استفاده شده است.

---

##  بخش اول: معرفی داده‌ها

###  داده‌های مورد استفاده:

* **دیتاست Iris** دارای 150 نمونه گل در سه گونه‌ی مختلف:

  1. Setosa
  2. Versicolor
  3. Virginica
* هر نمونه دارای 4 ویژگی عددی:

  * طول و عرض کاسبرگ
  * طول و عرض گلبرگ

###  پیش‌پردازش:

1. برچسب‌های متنی به اعداد صحیح (۰، ۱، ۲) تبدیل شده‌اند.
2. داده‌ها با استفاده از **StandardScaler** نرمال‌سازی شده‌اند (میانگین صفر و واریانس یک).
3. داده‌ها به دو بخش **آموزشی (۷۰٪)** و **آزمایشی (۳۰٪)** تقسیم شده‌اند.

---

##  بخش دوم: کاهش ابعاد

###  روش اول: **PCA** (تحلیل مؤلفه‌های اصلی)

* یک روش **بدون نظارت** (unsupervised) است که به دنبال یافتن جهاتی در فضای ویژگی‌هاست که در آن‌ها **بیشترین واریانس** داده‌ها حفظ شود.
* دو مؤلفه اصلی (PC1 و PC2) استخراج شد که بیشترین اطلاعات را نگه می‌دارند.

###  روش دوم: **LDA** (تحلیل تفکیک‌گر خطی)

* یک روش **نظارتی (supervised)** است که به دنبال جهاتی است که **جداکنندگی بین کلاس‌ها** را به حداکثر برساند.
* چون ۳ کلاس داریم، بیشترین تعداد مؤلفه‌های خطی خروجی می‌تواند **۲ مؤلفه** باشد (تعداد کلاس‌ها منهای یک).

---

##  بخش سوم: طبقه‌بندی

###  مدل مورد استفاده:

برای هر دو فضای کاهش‌یافته (PCA و LDA)، از مدل **رگرسیون لجستیک** استفاده شده است. این مدل، یک طبقه‌بند خطی ساده و قابل‌تفسیر است.

---

##  بخش چهارم: ارزیابی مدل‌ها

برای ارزیابی، از معیارهای زیر استفاده شده است:

| معیار                     | توضیح                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------ |
| **Accuracy**              | درصد نمونه‌های درست پیش‌بینی‌شده                                                     |
| **Confusion Matrix**      | ماتریسی که نشان می‌دهد مدل چند نمونه از هر کلاس را درست یا اشتباه طبقه‌بندی کرده است |
| **Classification Report** | شامل precision، recall و F1-score برای هر کلاس به‌صورت جداگانه                       |

###  ارزیابی مدل PCA:

* Accuracy: حدود 91٪
* طبقه‌بندی‌ها نسبتاً دقیق، اما برخی اشتباهات در تمایز بین کلاس‌های Versicolor و Virginica دیده می‌شود.

### x ارزیابی مدل LDA:

* Accuracy: حدود 97٪
* تفکیک بین کلاس‌ها بهتر انجام شده و دقت طبقه‌بندی بالاتر است.
* در اکثر اجراها، LDA عملکرد بهتری در مسائل طبقه‌بندی دارد، چرا که از برچسب کلاس‌ها در حین یادگیری استفاده می‌کند.

---

##  جمع‌بندی نهایی

| ویژگی                | PCA                 | LDA                           |
| -------------------- | ------------------- | ----------------------------- |
| نوع یادگیری          | بدون نظارت          | با نظارت                      |
| استفاده از برچسب     | ❌                   | ✅                             |
| معیار کاهش ابعاد     | بیشینه‌سازی واریانس | بیشینه‌سازی جدایی بین کلاس‌ها |
| مناسب برای طبقه‌بندی | نه همیشه            | بله، بسیار مناسب              |
| عملکرد نهایی         | حدود 91٪ دقت        | حدود 97٪ دقت                  |
