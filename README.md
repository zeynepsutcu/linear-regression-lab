# Linear Regression - Tek Değişkenli

Doğrusal regresyonun (linear regression) temel adımlarını gösteren küçük bir proje: model temsili, maliyet (cost) fonksiyonu ve gradient descent ile parametre öğrenme. Örnek veri: 1000 sqft'lik ev $300k, 2000 sqft'lik ev $500k.

## Kurulum

```
pip install -r requirements.txt
```

## Dosyalar

- `supervisedExample.py`: Model temsili, `f(x) = w*x + b` doğrusu, veriyi ve tahmin doğrusunu çizer, 1200 sqft için tahmin yapar.
- `cost_function.py`: `compute_cost` fonksiyonu, verilen w,b için modelin ne kadar hatalı olduğunu ölçer.
- `gradient_descent.py`: `compute_gradient` ve `gradient_descent` fonksiyonları, w,b parametrelerini otomatik olarak en düşük maliyete götürür, sonucu ve fiyat tahminlerini yazdırır.

## Çalıştırma

```
python supervisedExample.py
python cost_function.py
python gradient_descent.py
```

Grafik açan scriptlerde pencereyi kapattıkça bir sonraki adıma geçilir.

## Not

Kursun orijinal notebook'larında `lab_utils_uni.py` / `lab_utils.py` adında, kayan çubuklu (slider) ve contour grafik çizen yardımcı dosyalar kullanılıyor. Bu dosyalar kurs materyaliyle birlikte ayrıca indiriliyor ve burada yok; bu yüzden asıl matematiksel kod (cost, gradient, gradient descent) eksiksiz eklendi, sadece o yardımcı dosyaya bağımlı olan interaktif grafikler dışarıda bırakıldı.

## Kaynak

Bu kod, DeepLearning.AI ve Stanford Online tarafından hazırlanan "Machine Learning Specialization" (Coursera, Andrew Ng) eğitimindeki "Supervised Machine Learning: Regression and Classification" dersinin Hafta 1 "Optional Lab" materyallerinden uyarlanmıştır: Model Representation, Cost Function, Gradient Descent for Linear Regression.

https://www.coursera.org/specializations/machine-learning-introduction
