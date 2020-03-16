import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy.signal import convolve2d as conv2d
from scipy.linalg import circulant, norm
import cv2
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.neighbors import NearestNeighbors
from skimage import restoration, measure
from convolution_matrix import *
import os
import warnings
warnings.filterwarnings("ignore")


class Dip2:
    def __init__(self, alpha=2, q_size=5, k_size=9, n_neighbors=5, sigma=10, n_iter=5):
        self.q_size = q_size
        self.r_size = alpha * q_size
        self.k_size = k_size
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.n_iter = n_iter
        self.q_patches_num = None
        self.r_patches_num = None
        self.alpha = alpha

    def set_patches_limit(self, q_patches_num, r_patches_num):
        self.q_patches_num = q_patches_num
        self.r_patches_num = r_patches_num

    def generateLaplacian(self, size):
        C = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        padded_vec = np.zeros(size * size)
        d = abs(size - C.shape[1])
        index = 0
        for row in range(C.shape[0]):
            for col in range(C.shape[1]):
                padded_vec[index] = C[row, col]
                index += 1
            for i in range(d):
                padded_vec[index] = 0
                index += 1
        return circulant(padded_vec)

    def downsample_Rj(self, Rj, alpha):
        n = int(Rj.shape[0] ** 0.5)
        res = []
        for i in range(0, Rj.shape[0], alpha * n):
            for j in range(0, n, alpha):
                res.append(Rj[i + j, :])
        return np.array(res)

    def generateRjs(self, r_patches, curr_k):
        Rjs = []
        for r_patch in r_patches:
            Rj = generateRj(curr_k, r_patch)
            # r = convolution_as_maultiplication(curr_k, r_patch)
            # Rj = Rj[::alpha, ::]  # downsampling
            Rj = self.downsample_Rj(Rj, self.alpha)
            Rjs.append(Rj)
        return Rjs

    def calcWeightNominator(self, q, r_ds, sigma):
        return np.exp(-0.5 * (norm(q - r_ds) ** 2) / sigma ** 2)

    def calcWeightsMatrix(self, q_patches_vectorized, r_patches_ds_vectorized):
        # Fit r_alpha_patches into KNN
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, leaf_size=2, metric='euclidean').fit(
            r_patches_ds_vectorized)
        nbrs_weights = np.zeros((len(q_patches_vectorized), len(r_patches_ds_vectorized)))
        # for each q_patch find the n_neighbors nearest neighbors and calculate weight
        for i, q_patch_vectorized in enumerate(q_patches_vectorized):
            nbrs_indices = nbrs.kneighbors([q_patch_vectorized], return_distance=False)
            for idx in nbrs_indices[0]:
                nbrs_weights[i, idx] = self.calcWeightNominator(q_patch_vectorized, r_patches_ds_vectorized[idx],
                                                                self.sigma)

        # divide weights in a row by the total sum of the row
        nbrs_weights_sum = np.sum(nbrs_weights, axis=1)
        for i in range(len(nbrs_weights_sum)):
            if nbrs_weights_sum[i] == 0:
                nbrs_weights_sum[i] = 1
        nbrs_weights = np.divide(nbrs_weights, np.expand_dims(nbrs_weights_sum, axis=1))  # TODO: check axis
        return nbrs_weights

    def find_kernel(self, blurred_img):
        q_size, r_size, k_size = self.q_size, self.r_size, self.k_size
        q_patches = extract_patches_2d(blurred_img, (q_size, q_size), self.q_patches_num)
        r_patches = extract_patches_2d(blurred_img, (r_size, r_size), self.r_patches_num)
        curr_k = fftpack.fftshift(signal.unit_impulse((k_size, k_size)))
        C = self.generateLaplacian(k_size)
        CTC = C.T @ C
        Rjs = self.generateRjs(r_patches, curr_k)
        q_patches_vectorized = [matrix_to_vector(q_patch) for q_patch in q_patches]
        deconvolved_img = blurred_img
        plt.figure()
        for t in range(self.n_iter):
            plt.subplot(self.n_iter, 2, 2 * t + 1)
            plt.title('Iteration = {}'.format(t))
            plt.imshow(curr_k, cmap='gray')
            plt.subplot(self.n_iter, 2, 2 * t + 2)
            plt.imshow(deconvolved_img, cmap='gray')
            # calculate (r_j*curr_k) downsampled by alpha
            r_patches_ds_vectorized = []
            for i, r_patch in enumerate(r_patches):
                res = Rjs[i] @ (matrix_to_vector(curr_k))
                r_patches_ds_vectorized.append(res)

                # res = vector_to_matrix(res, [q_size, q_size])
                # res2 = conv2d(r_patch, curr_k, 'same', boundary='wrap')
                # res3 = conv2d(r_patch, curr_k, 'same', boundary='symm')
                # plt.figure()
                # plt.subplot(131)
                # plt.title('wrap boundary')
                # plt.imshow(res2, cmap='gray')
                # plt.subplot(132)
                # plt.title('symm boundary')
                # plt.imshow(res3, cmap='gray')
                # plt.subplot(133)
                # plt.title('Ours')
                # plt.imshow(res, cmap='gray')

            nbrs_weights = self.calcWeightsMatrix(q_patches_vectorized, r_patches_ds_vectorized)
            # calculating k_hat
            left_matrix_sum = np.zeros((k_size ** 2, k_size ** 2))
            right_vector_sum = np.zeros(k_size ** 2)
            for i in range(len(q_patches)):
                for j in range(len(r_patches)):
                    if nbrs_weights[i, j] == 0 or not math.isfinite(nbrs_weights[i, j]):
                        continue
                    left_matrix_sum += (nbrs_weights[i, j] / (self.sigma ** 2)) * (Rjs[j].T @ Rjs[j])
                    right_vector_sum += (nbrs_weights[i, j] / (self.sigma ** 2)) * (Rjs[j].T @ q_patches_vectorized[i])
                    # right_vector_sum += nbrs_weights[i, j] * (Rjs[j].T @ q_patches[i].reshape(q_size**2))

            left_matrix_sum += CTC  # TODO: Fix Laplacian
            curr_k = np.linalg.inv(left_matrix_sum) @ right_vector_sum
            curr_k = vector_to_matrix(curr_k, (k_size, k_size))
            curr_k = (1 / np.sum(curr_k)) * curr_k
            # TODO not sure about the missing factor (1 / (sigma ** 2))

            # display curr_k and the image  restored with the curr_k
            deconvolved_img = restoration.unsupervised_wiener(blurred_img, curr_k)
            deconvolved_img = deconvolved_img[0]
            # # plt.figure()
            # plt.subplot(121)
            # plt.title('Iteration = {}'.format(t))
            # plt.imshow(curr_k, cmap='gray')
            # plt.subplot(122)
            # plt.title('Iteration = {}'.format(t))
            # plt.imshow(deconvolved_img, cmap='gray')

        deconvolved_img = restoration.unsupervised_wiener(blurred_img, curr_k)
        deconvolved_img = deconvolved_img[0]
        # plt.figure()
        # plt.subplot(121)
        # plt.title('Final Kernel')
        # plt.imshow(curr_k, cmap='gray')
        #
        # plt.subplot(122)
        # plt.title('Final Image')
        # plt.imshow(deconvolved_img, cmap='gray')
        plt.subplot(self.n_iter, 2, 2 * self.n_iter - 1)
        plt.title('Iteration = {}'.format(self.n_iter - 1))
        plt.imshow(curr_k, cmap='gray')
        plt.subplot(self.n_iter, 2, 2 * self.n_iter)
        plt.imshow(deconvolved_img, cmap='gray')
        return curr_k, deconvolved_img


def Gaussian_Filter(high_res, sigma):
    BLUR_FILTER_SIZE = 11
    BLUR_FILTER_STD = sigma
    gaussian_img = cv2.GaussianBlur(high_res, (BLUR_FILTER_SIZE, BLUR_FILTER_SIZE), sigmaX=BLUR_FILTER_STD)
    return gaussian_img


def Sinc_Filter(high_res):
    SINC_SIZE = 11
    RANGE = 6
    x = np.linspace(-RANGE, RANGE, SINC_SIZE)
    xx = np.outer(x, x)
    sinc_func = np.sinc(xx)
    sinc_func = sinc_func / np.sum(sinc_func)
    # plt.figure()
    # plt.title('Sinc')
    # plt.imshow(sinc_func, cmap='gray')
    sinc_img = conv2d(high_res, sinc_func)
    return sinc_img


def main():
    high_res = cv2.imread("DIPSourceHW2.png", cv2.IMREAD_GRAYSCALE)
    high_res = (1 / 255) * high_res
    plt.figure()
    plt.title('Original image')
    plt.imshow(high_res, cmap='gray')
    results_folder = "./Results"
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    results_folder += "/"

    dip2 = Dip2(sigma=10, k_size=7)
    dip2.set_patches_limit(3000, 3000)

    gaussian_img = Gaussian_Filter(high_res, 10)
    gaussian_img = gaussian_img[::2, ::2]
    plt.figure()
    plt.imshow(gaussian_img, cmap='gray')
    plt.title('Gaussian_downsampled')
    plt.savefig(results_folder + 'Gaussian_downsampled.png')
    gaussian_k, gaussian_restored = dip2.find_kernel(gaussian_img)
    # plt.figure()
    # plt.imshow(gaussian_restored, cmap='gray')
    # plt.title('Gaussian_restored')
    # plt.savefig(results_folder + 'Gaussian_restored.png')
    plt.figure()
    plt.imshow(gaussian_k, cmap='gray')
    plt.title('Gaussian_k')
    plt.savefig(results_folder + 'Gaussian_k.png')
    dims = high_res.shape
    dims = dims[1], dims[0]
    upsampled_gaussian_image = cv2.resize(gaussian_restored, dsize=dims, interpolation=cv2.INTER_CUBIC)
    plt.figure()
    plt.imshow(upsampled_gaussian_image, cmap='gray')
    plt.title('Gaussian_restored')
    plt.savefig(results_folder + 'Gaussian_restored.png')
    gaussian_psnr = measure.compare_psnr(high_res, upsampled_gaussian_image)
    print("Gaussian PSNR: {}".format(gaussian_psnr))

    # create low-resolution image using sinc filter
    sinc_img = Sinc_Filter(high_res)
    sinc_img = sinc_img[::2, ::2]
    plt.figure()
    plt.imshow(sinc_img, cmap='gray')
    plt.title('Sinc_downsampled')
    plt.savefig(results_folder + 'Sinc_downsampled.png')
    sinc_k, sinc_restored = dip2.find_kernel(sinc_img)
    # plt.figure()
    # plt.imshow(sinc_restored, cmap='gray')
    # plt.title('Sinc_restored')
    # plt.savefig(results_folder + 'Sinc_restored.png')
    plt.figure()
    plt.imshow(sinc_k, cmap='gray')
    plt.title('Sinc_k')
    plt.savefig(results_folder + 'Sinc_k.png')
    dims = high_res.shape
    dims = dims[1], dims[0]
    upsampled_sinc_image = cv2.resize(sinc_restored, dsize=dims, interpolation=cv2.INTER_CUBIC)
    plt.figure()
    plt.imshow(upsampled_sinc_image, cmap='gray')
    plt.title('Sinc_restored')
    plt.savefig(results_folder + 'Sinc_restored.png')
    sinc_psnr = measure.compare_psnr(high_res, upsampled_sinc_image)
    print("Sinc PSNR: {}".format(sinc_psnr))

    wrong_gaussian_kernel = restoration.unsupervised_wiener(gaussian_img, sinc_k)
    wrong_gaussian_kernel = wrong_gaussian_kernel[0]
    wrong_gaussian_kernel = cv2.resize(wrong_gaussian_kernel, dsize=dims, interpolation=cv2.INTER_CUBIC)
    plt.figure()
    plt.imshow(wrong_gaussian_kernel, cmap='gray')
    plt.title('Gaussian_restored_wrong')
    plt.savefig(results_folder + 'Gaussian_restored_wrong.png')
    gaussian_psnr = measure.compare_psnr(high_res, wrong_gaussian_kernel)
    print("Gaussian_Wrong PSNR: {}".format(gaussian_psnr))

    wrong_sinc_kernel = restoration.unsupervised_wiener(sinc_img, gaussian_k)
    wrong_sinc_kernel = wrong_sinc_kernel[0]
    wrong_sinc_kernel = cv2.resize(wrong_sinc_kernel, dsize=dims, interpolation=cv2.INTER_CUBIC)
    plt.figure()
    plt.imshow(wrong_sinc_kernel, cmap='gray')
    plt.title('Sinc_restored_wrong')
    plt.savefig(results_folder + 'Sinc_restored_wrong.png')
    sinc_psnr = measure.compare_psnr(high_res, wrong_sinc_kernel)
    print("Sinc_Wrong PSNR: {}".format(sinc_psnr))

    # plt.show()

    # max_psnr = 0
    # max_sigma = 0.1
    # for sigma in np.linspace(0.1, 100, 200):
    #     dip2 = Dip2(sigma=sigma, k_size=7)
    #     dip2.set_patches_limit(3000, 3000)
    #     sinc_k, sinc_restored = dip2.find_kernel(sinc_img)
    #     dims = high_res.shape
    #     dims = dims[1], dims[0]
    #     upsampled_sinc_image = cv2.resize(sinc_restored, dsize=dims, interpolation=cv2.INTER_CUBIC)
    #     plt.figure()
    #     plt.imshow(upsampled_sinc_image, cmap='gray')
    #     plt.title('upsampled_sinc_image')
    #     plt.savefig('upsampled_sinc_image.png')
    #     plt.close('all')
    #     sinc_psnr = measure.compare_psnr(high_res, upsampled_sinc_image)
    #     if sinc_psnr > max_psnr:
    #         max_psnr = sinc_psnr
    #         max_sigma = sigma
    # print("max_psnr={},max_sigma={}".format(max_psnr, max_sigma))
    # max_psnr=17.396481227445626,max_sigma=15.16030150753769


if __name__ == "__main__":
    main()
