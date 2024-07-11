loaded_generator = tf.keras.models.load_model('generator_modelDCGAN.keras')
random_noise = tf.random.normal([1, 100])
generated_images = loaded_generator(random_noise, training=False)
plt.imshow(generated_images[0, :,:,0], cmap='gray');
