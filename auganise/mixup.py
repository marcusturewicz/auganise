# mixup: Beyond Empirical Risk Minimization
# arXiv:1710.09412

import tensorflow as tf

def _sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def _mixup(dataset1, dataset2, alpha):
    # Unpack
    images1, labels_one = dataset1
    images2, labels_two = dataset2
    batch_size = tf.shape(images1)[0]

    # mixup
    lambda_ = _sample_beta_distribution(batch_size, alpha, alpha)
    x_lambda = tf.reshape(lambda_, (batch_size, 1, 1, 1))
    y_lambda = tf.reshape(lambda_, (batch_size, 1))

    images = images1 * x_lambda + images2 * (1 - x_lambda)
    labels = labels_one * y_lambda + labels_two * (1 - y_lambda)
    return (images, labels)

def augment(dataset, alpha):
    num_batches = dataset.cardinality().numpy()
    dataset2 = dataset.take(num_batches)
    dataset_combined = tf.data.Dataset.zip((dataset, dataset2)) 
    dataset_mixup = dataset_combined.map(
        lambda dataset1, dataset2: _mixup(dataset1, dataset2, alpha),
            num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset_mixup  

    
