from PIL import Image
import numpy as np
import settings
import os

def PIL2array(img):
    if type(img.getdata()[1]) == int:
        img = img.convert('RGB')
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[0], img.size[1], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def open_img(src):
    im = Image.open(src)
    if settings.resize:
        im = im.resize(settings.size)
    try:
        PIL2array(im)
    except ValueError:
        print(im.getdata()[1] ,src)
        exit()
    return PIL2array(im)

def save_img(src, data):
    im = array2PIL(data, (data.shape[0], data.shape[1]))
    im.save(src)

def vectorize(y, size=62):
    v = np.zeros(size)
    v[y-1] = 1
    return v

def load_bmps(src='Img/GoodImg'):
    xs = []
    ys = []
    src = os.path.join(src, 'Bmp')
    samples = [n for n in os.listdir(src) if n.startswith('Sample')]
    size = len(samples)
    for s in samples:
        y_pre = s[::-1][:3][::-1]
        y = vectorize(int(y_pre),size)
        ls = [n for n in os.listdir(os.path.join(src, s)) if os.path.isfile(os.path.join(src, s, n))]
        for l in ls:
            x = open_img(os.path.join(src, s, l))
            xs.append(x)
            ys.append(y)
    x = np.array(xs)
    y = np.array(ys)
    return (x, y)

def load_chars74k(src='Img'):
    x, y = load_bmps(os.path.join(src,'GoodImg'))
    if not settings.only_good_imgs:
        x_bad, y_bad = load_bmps(os.path.join(src,'BadImag'))
        x = np.concatenate((x, x_bad))
        y = np.concatenate((y, y_bad))
    rate = settings.validation_rate
    test = np.random.choice(range(x.shape[0]), int(x.shape[0] * rate))
    x_train = [x[i] for i in set(range(x.shape[0])) - set(test)]
    x_test = [x[i] for i in test]
    x_train = sorted(x_train, key = lambda i : len(i))
    y_train = [y[i] for i in set(range(x.shape[0])) - set(test)]
    y_test = [x[i] for i in test]
    y_train = sorted(y_train, key = lambda i : len(i))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return (x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_chars74k()
    print(x_train.shape, y_train.shape)