import numpy as np

def custom_histogram(one_d_img, rangemax=256):
    res = np.zeros(rangemax)
    for i in range(len(one_d_img)):
        res[one_d_img[i]] += 1
    return res


def linear_form(Input_px, A, B, C, D):
    assert(not(A == B))
    return ((Input_px - A) * ((D-C)/(B-A))) + C


# ---------------------------------- Linear stretching histogram ----------------------------------
def linear_hist(img, C=0, D=255):
    area = img.size
    oneD_img = np.uint8(img.ravel().copy())
    hist = custom_histogram(oneD_img)
    # find min and max intensity with 5% area reduce
    A = min(oneD_img)
    B = max(oneD_img)
    limit = int((area * 0.05))

    # historgram 5% shrink 
    while limit > 0:
        if (hist[A] < hist[B]):
            limit -= hist[A]
            A += 1
        else:
            limit -= hist[B]
            B -= 1  
    Temp = np.clip(oneD_img, A, B)
    # apply linear stretch formula for every pixel
    Result = np.uint8(np.apply_along_axis(linear_form, 0, Temp, A, B, C, D))
    Result = Result.reshape(img.shape)
    return Result


# ---------------------------------- histogram equalization ----------------------------------
def histogram_equalization(img):
    area = img.size   
    oneD_img = np.uint8(img.ravel().copy())
   
    hist, _ = np.histogram(oneD_img, bins=np.arange(256))

    # calc cumulative sum
    cumul_sum = hist.cumsum()
  
    norm_cs = (cumul_sum - cumul_sum.min()) * 255
    N = cumul_sum.max() - cumul_sum.min()
    cumul_sum = (norm_cs / N).astype('uint8')

    # get equalization histogram
    Result = cumul_sum[oneD_img]
    Result = Result.reshape(img.shape)
    return Result