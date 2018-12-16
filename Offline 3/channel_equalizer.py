import numpy as np
import math
# from scipy.stats import multivariate_normal


class ChannelEqualizer:
    def __init__(self, coefficients, noise_mean, noise_variance):
        self.clusters_means = []
        self.prior_probas = []
        self.coefficients = coefficients
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.clusters_covariances = []
        self.num_of_clusters = 8

        self.transition_probas = []
        for i in range(self.num_of_clusters):
            temp = []
            for j in range(self.num_of_clusters):
                if (i == 0 or i == 1) and (j == 0 or j == 4):
                    temp.append(0.5)
                elif (i == 2 or i == 3) and (j == 1 or j == 5):
                    temp.append(0.5)
                elif (i == 4 or i == 5) and (j == 2 or j == 6):
                    temp.append(0.5)
                elif (i == 6 or i == 7) and (j == 3 or j == 7):
                    temp.append(0.5)
                else:
                    temp.append(0)
            self.transition_probas.append(temp)

    @staticmethod
    def construct_covariance_matrix(data):
        means = np.mean(data, axis=0)
        n = data.shape[0]
        temp = np.matrix(data[0, :]) - np.matrix(means)
        temp_t = temp.transpose()
        covariance_matrix = temp_t * temp
        for i in range(1, n):
            temp = np.matrix(data[i, :]) - np.matrix(means)
            temp_t = temp.transpose()
            covariance_matrix = covariance_matrix + (temp_t * temp)
        covariance_matrix /= n
        return covariance_matrix

    @staticmethod
    def multivariate_normal(x, means, covariance_mat):
        d = len(x)
        temp1 = math.sqrt(math.pow(2 * math.pi, d) * math.fabs(np.linalg.det(covariance_mat)))
        temp2 = np.matrix(x) - np.matrix(means)
        temp3 = -0.5 * temp2 * np.linalg.inv(covariance_mat) * temp2.transpose()
        temp4 = math.exp(temp3)
        result = temp4 / temp1
        return result

    def distance_proba(self, x, w_after, w_before):
        if self.transition_probas[w_before][w_after] == 0:
            return math.inf
        elif w_before == -1:
            d = self.prior_probas[w_after] * self.multivariate_normal(x, self.clusters_means[w_after],
                                                                      self.clusters_covariances[w_after])
            return math.log(d, math.e)
        else:
            d = self.transition_probas[w_before][w_after]*self.multivariate_normal(x, self.clusters_means[w_after],
                                                                                   self.clusters_covariances[w_after])
            return math.log(d, math.e)

    def D_max(self, wik, x, k, from_list, D):
        if k == 1:
            return self.distance_proba(x[k], wik, -1)
        if D[0][k - 1] == -1:
            max_value = self.D_max(0, x, k - 1, from_list, D) + self.distance_proba(x[k], wik, 0)
        else:
            max_value = D[0][k - 1] + self.distance_proba(x[k], wik, 0)
        max_from = 0
        for wik_1 in range(1, self.num_of_clusters):
            if D[wik_1][k - 1] == -1:
                value = self.D_max(0, x, k - 1, from_list, D) + self.distance_proba(x[k], wik, 0)
            else:
                value = D[wik_1][k - 1] + self.distance_proba(x[k], wik, 0)
            if value > max_value:
                max_value = value
                max_from = wik_1
        from_list[wik][k] = max_from
        return max_value

    def train(self, I):
        coefficients = self.coefficients
        noise_mean = self.noise_mean
        noise_variance = self.noise_variance

        num_of_clusters = self.num_of_clusters

        noise_list = np.random.normal(noise_mean, noise_variance, len(I))

        x = len(I)*[0]

        clusters = []
        clusters_means = []
        prior_probas = []
        clusters_covariances = []

        for j in range(num_of_clusters):
            clusters.append([])

        for k in range(1, len(I)):
            if k == 1:
                x[k] = coefficients[0] * I[k] + noise_list[k]
                cluster_no = I[k]*4
            else:
                x[k] = coefficients[0]*I[k] + coefficients[1]*I[k - 1] + noise_list[k]
                cluster_no = I[k]*4 + I[k - 1]*2 + I[k - 2]*1

            clusters[cluster_no].append([x[k], x[k - 1]])

        total_datapoints = 0
        for j in range(num_of_clusters):
            print(j)
            cluster_size = float(len(clusters[j]))
            total_datapoints += cluster_size
            prior_probas.append(cluster_size)

            clusters[j] = np.array(clusters[j])
            print(clusters[j].shape)
            clusters_means.append(np.mean(clusters[j], axis=0))
            clusters_covariances.append(self.construct_covariance_matrix(clusters[j]))

        prior_probas = [x / total_datapoints for x in prior_probas]
        print(prior_probas)
        self.clusters_means = clusters_means
        self.prior_probas = prior_probas
        self.clusters_covariances = clusters_covariances

    def predict(self, I):
        coefficients = self.coefficients
        noise_mean = self.noise_mean
        noise_variance = self.noise_variance

        noise_list = np.random.normal(noise_mean, noise_variance, len(I))

        x = len(I)*[0]
        y = len(I)*[0]
        X = [0]

        D = [[-1 for _ in range(len(x))] for _ in range(self.num_of_clusters)]
        from_list = [[-1 for _ in range(len(x))] for _ in range(self.num_of_clusters)]

        for k in range(1, len(I)):
            x[k] = coefficients[0]*I[k] + coefficients[1]*I[k - 1] + noise_list[k]
            X.append([x[k], x[k - 1]])

            for i in range(self.num_of_clusters):
                print(k, i)
                D[i][k] = self.D_max(i, X, k, from_list, D)

        D_max = D[0][len(I) - 1]
        cluster_max = 0
        for i in range(1, self.num_of_clusters):
            D_val = D[i][len(I) - 1]
            if D_val > D_max:
                D_max = D_val
                cluster_max = i

        for k in range(len(I) - 1, 0):
            if cluster_max in range(0, 4):
                y[k] = 0
            else:
                y[k] = 1
            cluster_max = from_list[cluster_max][k]
        return y
