
###  مقدمه:

الگوریتم Monte Carlo به‌صورت اپیزودیک عمل می‌کند، یعنی تا پایان هر اپیزود صبر می‌کند و سپس مقادیر پاداش نهایی را برای به‌روزرسانی ارزش‌ها استفاده می‌کند.

در این پروژه از روش First-Visit Monte Carlo برای یادگیری Q(state, action) استفاده شده است.

---

###  تنظیمات محیط:

* ساختار GridWorld ساده
* حرکت در چهار جهت
* پاداش +1 برای رسیدن به هدف، -1 برای برخورد با دیوار، و -0.01 برای هر حرکت

---

###  پارامترها:

| پارامتر      | مقدار |
| ------------ | ----- |
| γ (تخفیف)    | 0.99  |
| ε (اکتشاف)   | 0.1   |
| تعداد اپیزود | 1000  |

---

###  روش الگوریتم:

* عامل اپیزودهای تصادفی تولید می‌کند
* برای هر state-action در اپیزود، اگر اولین‌بار باشد که دیده می‌شود، پاداش G از آن نقطه تا پایان محاسبه شده و در میانگین Q لحاظ می‌شود
* سیاست با استفاده از Q و روش epsilon-greedy انتخاب می‌شود

---

###  ارزیابی:

پس از یادگیری، سیاست حاصل در 100 اپیزود تست بدون اکتشاف بررسی شد. درصد موفقیت اعلام شد:

```
 Monte Carlo Success rate: XX.XX% in 100 test episodes.
```

---


* الگوریتم مونت‌کارلو برای محیط‌های اپیزودیک ساده مناسب است
* نسبت به Q-learning وابسته به پاداش‌های نهایی است و نیاز به صبر تا پایان اپیزود دارد
* در محیط‌های کوچک عملکرد خوبی دارد اما در محیط‌های بزرگ نیاز به بهینه‌سازی دارد
