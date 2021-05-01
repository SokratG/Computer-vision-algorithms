import numpy as np

def get_spectrum(dft_img):
    spectrum_img = np.log10(1 + np.absolute(dft_img))
    return spectrum_img
    

# ---------------------------------- DFT ----------------------------------
def DFT_img(img, use_spectrum=False): # very slow!
    img_h, img_w = img.shape
    dft_img = np.zeros((img_h - 1, img_w - 1), dtype=np.complex)
    center_img = img.copy()
    
    for y in range(img_h):
        for x in range(img_w):
            center_img[y, x] *= np.power(-1, x + y)
    
    exp_coef = -2 * np.pi * 1j
   	# O((N*M)^2)
    for u in range(img_h - 1):
        for v in range(img_w - 1):
            acc_val = 0.0
            r = (exp_coef * u) / img_w
            c = (exp_coef * v) / img_h
            for y in range(img_h - 1):
                for x in range(img_w - 1): 
                    e_complex = np.exp((x * r) + (y * c))
                    if (use_spectrum):
                        acc_val += center_img[y, x] * e_complex
                    else:
                        acc_val += img[y, x] * e_complex
            dft_img[u, v] = acc_val
            
    # optional:
    power_set_factor = 1.0 / (img_h * img_w)
    dft_img *= power_set_factor
    
    if (use_spectrum):
        return get_spectrum(dft_img)
    else:
        return dft_img

    

# ---------------------------------- Inverse DFT ----------------------------------
def IDFT_img(dft_img):
    import utilCV
    img_h, img_w = img.shape
    res_img = np.zeros((img_h - 1, img_w - 1), dtype=np.float)


    exp_coef = 1j * 2 * np.pi
   	# O((N*M)^2)
    for x in range(img_h - 1):
        for y in range(img_w - 1):
            acc_val = 0.0
            r = (exp_coef * x) / img_w
            c = (exp_coef * y) / img_h
            for u in range(img_h - 1):
                for v in range(img_w - 1): 
                    e_complex = np.exp((u * r) + (v * c))
                    acc_val += dft_img[u, v] * e_complex #
            res_img[y, x] = acc_val

    # optional:
    # power_set_factor = 1.0 / (img_h * img_w)
    # res_img *= power_set_factor

    return utilCV.color_renormalize(res_img)