import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resmi grayscale olarak yükle
path = "C:/Users/Ibrahim/Desktop/hucre.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Resim yüklenmediyse hata mesajı ver ve çık
if image is None:
    print("Hata: Resim yüklenemedi.")

# Keskinleştirmek için bir kernel tanımla
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])

# Kerneli grayscale resme uygula
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

#Gaussian bulanıklık (blur) uygulama
blurred_image = cv2.GaussianBlur(sharpened_image, (29, 29), 0)

#Adaptif eşikleme işlemi 
adaptive_threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

#Morfolojik işlem (erozyon ve genişletme) ve ikili görüntü işleme işlemlerini içerir.
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(adaptive_threshold_image, kernel, iterations=1)
bitwise_not = cv2.bitwise_not(eroded_image) 
Idil = cv2.dilate(bitwise_not,kernel,iterations=1)

#Connected components analysis
thresh, im_tr = cv2.threshold(Idil,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
nb_components, comp, stats, centroids = cv2.connectedComponentsWithStats(im_tr, connectivity=8)

#Canny kenar tespiti ve ardından bu kenarların bağlı bileşen analizi
kenarlar = cv2.Canny(im_tr, threshold1=25, threshold2=100)
_, labels = cv2.connectedComponents(kenarlar)

#Bağlı bileşenlerin etiketlerini kullanarak farklı bağlı bileşenlere farklı renkler atama

colored_labels = np.zeros_like(labels, dtype=np.uint8)
for label in range(1, np.max(labels)+1):
    colored_labels[labels == label] = label * 10  # Farklı renkler için farklı değerler kullanabiliriz

#Bağlı bileşenlerin geometrik özelliklerini analiz etmek

for label in range(1, np.max(labels)+1):
    # Bileşenin piksel sayısını hesaplayın (Alan)
    alan = np.sum(labels == label)
    
    # Bileşenin momentlerini hesaplayarak yönlendirme (Orientation) bulun
    moments = cv2.moments(np.uint8(labels == label))
    yon = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    
    # Dairesellik (Circularity) hesaplamak için dairesel birliği hesaplayın
    dairesellik = 4 * np.pi * alan / (moments['m00'] ** 2)
    
    # Sonuçları yazdırın
    print(f'Bileşen {label}: Alan = {alan}, Yön = {yon}, Dairesellik = {dairesellik}')
    
print(f'###################################################################')

# Her bileşen için öznitelikleri hesaplama
for label in range(1, np.max(labels)+1):
    alan = np.sum(labels == label) 
    kontur, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cevre = cv2.arcLength(kontur[0], True)   
    kompaktlik = 4 * np.pi * alan / (cevre ** 2)    
    print(f'Bileşen {label}: Alan = {alan}, Çevre = {cevre}, Kompaktlık = {kompaktlik}')
    

plt.subplot(3, 4, 1)
plt.imshow(image, cmap="gray")
plt.title('1.Orjinal')

plt.subplot(3, 4, 2)
plt.imshow(sharpened_image, cmap="gray" )
plt.title('2.Sharpened')

plt.subplot(3, 4, 3)
plt.imshow(adaptive_threshold_image, cmap="gray")
plt.title('3.Adaptive Threshold')

plt.subplot(3, 4, 4)
plt.imshow(eroded_image, cmap="gray" )
plt.title('4.Eroded')

plt.subplot(3, 4, 5)
plt.imshow(bitwise_not, cmap="gray" )
plt.title('5.Bitwise')

plt.subplot(3, 4, 6)
plt.imshow(Idil, cmap="gray" )
plt.title('6.Dilate')

plt.subplot(3, 4, 7)
plt.imshow(colored_labels,cmap="nipy_spectral")
plt.title('7.Renklendirilmiş Bileşenler')

plt.subplot(3, 4, 8)
plt.imshow(comp,cmap="nipy_spectral")
plt.title('8.Sonuç')





