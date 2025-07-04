

##  گزارش پروژه: طبقه‌بندی قارچ‌ها با استفاده از درخت تصمیم

###  معرفی داده‌ها

داده‌های مورد استفاده در این پروژه، مجموعه داده‌ی معروف "Mushroom Dataset" از مخزن UCI Machine Learning Repository است. این مجموعه داده شامل 8124 نمونه از انواع مختلف قارچ‌ها می‌باشد که با ویژگی‌های ظاهری و زیستی مانند شکل کلاهک، رنگ، بوی قارچ و سایر مشخصات توصیف شده‌اند. هدف این است که مشخص کنیم آیا یک قارچ **قابل خوردن (edible)** یا **سمی (poisonous)** است.

تعداد ویژگی‌ها: 22 ویژگی توصیفی (همگی کیفی)
برچسب خروجی: متغیر `class` با دو مقدار `e` (خوراکی) و `p` (سمی)

---

###  مراحل انجام کار

#### 1. **بارگذاری داده‌ها**

داده‌ها مستقیماً از آدرس اینترنتی UCI بارگذاری شدند. برای هر ستون نام مشخصی در نظر گرفته شد تا کار پردازش ساده‌تر شود.

#### 2. **پیش‌پردازش**

* مقدارهای گمشده (مانند مقدار '?' در ستون `stalk-root`) حذف شدند.
* تمامی ویژگی‌ها کیفی (categorical) بودند، بنابراین از **LabelEncoder** برای تبدیل آن‌ها به عدد استفاده کردیم تا برای مدل قابل استفاده شوند.
* داده‌ها به دو بخش آموزش (70%) و آزمون (30%) تقسیم شدند.

#### 3. **آموزش مدل درخت تصمیم**

از الگوریتم **Decision Tree Classifier** از کتابخانه `scikit-learn` استفاده کردیم. معیار تقسیم گره‌ها بر اساس **آنتروپی (Entropy)** در نظر گرفته شد. مدل آموزش داده شد و سپس روی داده‌های آزمون اعمال شد.

#### 4. **ترسیم درخت تصمیم**

با استفاده از تابع `plot_tree` از کتابخانه `matplotlib`، ساختار کامل درخت تصمیم رسم شد تا بتوان مراحل تصمیم‌گیری مدل را مشاهده کرد.

---

###  ارزیابی مدل

برای ارزیابی عملکرد مدل، از چندین معیار استاندارد استفاده کردیم که همگی بر اساس ماتریس درهم‌ریختگی (Confusion Matrix) تعریف می‌شوند.

#### مقادیر به‌دست‌آمده:

| معیار                         | مقدار  |
| ----------------------------- | ------ |
| دقت (Accuracy)                | 1.0000 |
| دقت مثبت (Precision)          | 1.0000 |
| حساسیت (Sensitivity / Recall) | 1.0000 |
| ویژگی‌مندی (Specificity)      | 1.0000 |
| F1 Score                      | 1.0000 |
| شاخص Youden (SPB)             | 1.0000 |

####  تعریف معیارها:

* **ماتریس درهم‌ریختگی (Confusion Matrix):**
  شامل 4 مقدار است:

  * TP: تعداد درست پیش‌بینی شده‌های مثبت (سمی‌های درست)
  * TN: تعداد درست پیش‌بینی شده‌های منفی (خوراکی‌های درست)
  * FP: مثبت‌های کاذب (خوراکی‌های اشتباه پیش‌بینی شده به عنوان سمی)
  * FN: منفی‌های کاذب (سمی‌هایی که به اشتباه خوراکی تشخیص داده شدند)

* **Accuracy (دقت کلی)**:
  نسبت تعداد پیش‌بینی‌های صحیح به کل نمونه‌ها

  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

* **Precision (دقت مثبت‌ها)**:
  از بین مواردی که به عنوان سمی پیش‌بینی شدند، چند مورد واقعاً سمی بوده‌اند؟

  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$

* **Sensitivity / Recall (حساسیت)**:
  از کل موارد سمی واقعی، چند مورد شناسایی شدند؟

  $$
  \text{Sensitivity} = \frac{TP}{TP + FN}
  $$

* **Specificity (ویژگی‌مندی)**:
  از کل خوراکی‌های واقعی، چند مورد درست شناسایی شدند؟

  $$
  \text{Specificity} = \frac{TN}{TN + FP}
  $$

* **F1 Score**:
  میانگین هماهنگ دقت و حساسیت

  $$
  \text{F1} = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
  $$

* **شاخص Youden (SPB / J-index)**:
  معیار ترکیبی از حساسیت و ویژگی‌مندی برای ارزیابی کلی مدل

  $$
  \text{SPB} = Sensitivity + Specificity - 1
  $$

---

###  نتیجه‌گیری

با توجه به مقادیر ارزیابی به‌دست‌آمده، مدل درخت تصمیم در این مسئله عملکردی **کامل (Perfect)** دارد و تمام نمونه‌های آزمون را به درستی طبقه‌بندی کرده است. این موضوع به دلیل ماهیت داده‌ها و ویژگی‌های کاملاً متمایز بین قارچ‌های سمی و خوراکی است.

در نتیجه می‌توان گفت الگوریتم Decision Tree برای این نوع داده‌ها انتخابی بسیار مناسب بوده و با پردازش مناسب می‌تواند در کاربردهای واقعی نیز مورد استفاده قرار گیرد.

