from io import BytesIO

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from jaxtyping import Array, Float, Int, PRNGKeyArray
from PIL import Image

from jaxonlayers.functions.embedding import sinusoidal_embedding
from jaxonlayers.functions.utils import summarize_model

# Load the image using Pillow
# img = Image.open("cat.jpg")

# # Convert to numpy array
# img_array = np.array(img).astype(np.float64)


BETA = 10
T = 100000

# xs = [img_array]
# for t in range(1, T):
#     slightly_noisier_image = xs[t - 1] + BETA * np.random.standard_normal(
#         size=img_array.shape
#     )
#     xs.append(slightly_noisier_image)

# # Display each image in xs using matplotlib
# fig, axes = plt.subplots(2, 10, figsize=(30, 6))

# # First row: first 10 images
# for i in range(10):
#     # Clip values to valid image range [0, 255] and convert to uint8
#     display_img = np.clip(xs[i], 0, 255).astype(np.uint8)
#     axes[0, i].imshow(display_img)
#     axes[0, i].set_title(f"Image {i} (t={i})")
#     axes[0, i].axis("off")

# # Second row: last 10 images
# for i in range(10):
#     img_index = len(xs) - 10 + i
#     # Clip values to valid image range [0, 255] and convert to uint8
#     display_img = np.clip(xs[img_index], 0, 255).astype(np.uint8)
#     axes[1, i].imshow(display_img)
#     axes[1, i].set_title(f"Image {img_index} (t={img_index})")
#     axes[1, i].axis("off")

# plt.tight_layout()
# plt.show()

# X_T = np.random.normal(img_array, T * np.sqrt(BETA))
# expected = np.random.standard_normal(size=X_T.shape)
# expected_min = np.min(expected)
# expected_max = np.max(expected)
# expected = 255 * (expected - expected_min) / (expected_max - expected_min)


# # using the direct formula
# # assuming constant beta
# BETA = 0.9
# X_T_DIRECT = np.sqrt(((1 - BETA) ** T)) + np.sqrt(
#     (1 - (1 - BETA) ** T)
# ) * np.random.standard_normal(size=img_array.shape)
# # Normalize X_T_DIRECT to [0, 255] range
# X_T_DIRECT_min = np.min(X_T_DIRECT)
# X_T_DIRECT_max = np.max(X_T_DIRECT)
# X_T_DIRECT = 255 * (X_T_DIRECT - X_T_DIRECT_min) / (X_T_DIRECT_max - X_T_DIRECT_min)


# # Display all three images side by side using matplotlib
# fig, axes = plt.subplots(1, 3, figsize=(30, 8))

# # First image: X_T
# display_img1 = np.clip(X_T, 0, 255).astype(np.uint8)
# axes[0].imshow(display_img1)
# axes[0].set_title(f"X_T (t={T})")
# axes[0].axis("off")

# # Second image: expected
# display_img2 = np.clip(expected, 0, 255).astype(np.uint8)
# axes[1].imshow(display_img2)
# axes[1].set_title(f"Expected")
# axes[1].axis("off")

# # Third image: X_T_DIRECT
# display_img3 = np.clip(X_T_DIRECT, 0, 255).astype(np.uint8)
# axes[2].imshow(display_img3)
# axes[2].set_title(f"X_T_DIRECT")
# axes[2].axis("off")

# plt.tight_layout()
# plt.show()


class DataFrameDataSource:
    def __init__(self, data_list: list[dict]):
        self._data = data_list

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, record_key: int) -> dict:
        return self._data[record_key]


class MNISTPreprocessor(grain.transforms.Map):
    def map(self, element):
        image_dict = element["image"]
        label = element["label"]

        image_bytes = image_dict["bytes"]
        pil_image = Image.open(BytesIO(image_bytes))

        image_array = np.array(pil_image, dtype=np.float32)
        image_array = image_array / 255.0
        width, height = image_array.shape
        image_array = image_array.reshape(1, width, height)

        return {
            "image": jnp.array(image_array),
            "label": jnp.array(label, dtype=jnp.int32),
        }


def load_mnist_from_hf(split="train"):
    splits = {
        "train": "mnist/train-00000-of-00001.parquet",
        "test": "mnist/test-00000-of-00001.parquet",
    }

    url = "hf://datasets/ylecun/mnist/" + splits[split]
    table = pq.read_table(url)
    return table.to_pylist()


class ConvBlock(eqx.Module):
    time_mlp: eqx.nn.Linear
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_size: int,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jax.random.split(key, 3)
        self.time_mlp = eqx.nn.Linear(time_embedding_size, out_channels, key=key1)
        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, key=key2
        )
        self.conv2 = eqx.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, key=key3
        )

    def __call__(
        self,
        x: Float[Array, "channel width height"],
        t: Float[Array, "time_embedding_size"],
    ) -> Float[Array, "channel width height"]:
        h = self.conv1(x)
        h = jax.nn.silu(h)

        time_emb = self.time_mlp(t)
        time_emb = jax.nn.silu(time_emb)
        time_emb = time_emb[:, None, None]

        h = h + time_emb

        h = self.conv2(h)
        h = jax.nn.silu(h)

        return h


class DiffusionModel(eqx.Module):
    time_mlp: eqx.nn.MLP
    time_embedding_size: int

    initial_conv: eqx.nn.Conv2d

    down1: ConvBlock
    down2: ConvBlock
    pool: eqx.nn.MaxPool2d

    bottleneck: ConvBlock

    up1: eqx.nn.ConvTranspose2d
    up_conv1: ConvBlock
    up2: eqx.nn.ConvTranspose2d
    up_conv2: ConvBlock

    final_conv: eqx.nn.Conv2d

    def __init__(self, time_embedding_size: int, *, key: PRNGKeyArray):
        self.time_embedding_size = time_embedding_size

        keys = jax.random.split(key, 10)

        self.initial_conv = eqx.nn.Conv2d(1, 32, kernel_size=3, padding=1, key=keys[0])

        self.down1 = ConvBlock(32, 64, time_embedding_size, key=keys[1])
        self.down2 = ConvBlock(64, 128, time_embedding_size, key=keys[2])
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(128, 256, time_embedding_size, key=keys[3])

        self.up1 = eqx.nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2, key=keys[4]
        )
        self.up_conv1 = ConvBlock(256, 128, time_embedding_size, key=keys[5])
        self.up2 = eqx.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, key=keys[6])
        self.up_conv2 = ConvBlock(128, 64, time_embedding_size, key=keys[7])

        self.final_conv = eqx.nn.Conv2d(64, 1, kernel_size=1, key=keys[8])

        self.time_mlp = eqx.nn.MLP(
            in_size=time_embedding_size,
            out_size=time_embedding_size,
            width_size=time_embedding_size * 4,
            depth=2,
            key=keys[9],
        )

    def __call__(
        self, x_t: Float[Array, "channel height width"], t: Int[Array, ""]
    ) -> Float[Array, "channel height width"]:
        time_embeddings = sinusoidal_embedding(t, self.time_embedding_size)
        time = self.time_mlp(time_embeddings)

        h = self.initial_conv(x_t)

        skip1 = self.down1(h, time)
        h = self.pool(skip1)

        skip2 = self.down2(h, time)
        h = self.pool(skip2)

        h = self.bottleneck(h, time)

        h = self.up1(h)
        h = jnp.concatenate([h, skip2], axis=0)
        h = self.up_conv1(h, time)

        h = self.up2(h)
        h = jnp.concatenate([h, skip1], axis=0)
        h = self.up_conv2(h, time)

        output = self.final_conv(h)
        return output


time_embedding_size = 32


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return jnp.linspace(beta_start, beta_end, timesteps)


TIMESTEPS = 200

betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1.0 - betas
alpha_bars = jnp.cumprod(alphas, axis=0)

sqrt_alpha_bars = jnp.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = jnp.sqrt(1.0 - alpha_bars)


def loss_fn(
    model: DiffusionModel, x_0: Float[Array, "channel height width"], key: PRNGKeyArray
) -> Float[Array, ""]:
    t_key, noise_key = jax.random.split(key)
    t = jax.random.randint(t_key, shape=(), minval=0, maxval=TIMESTEPS)
    noise = jax.random.normal(noise_key, shape=x_0.shape)

    sqrt_alpha_bar_t = sqrt_alpha_bars[t]
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bars[t]

    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    predicted_noise = model(x_t, t)

    loss = jnp.mean((noise - predicted_noise) ** 2)
    return loss


# Create a new, batched version of the loss function using vmap
batched_loss_fn = eqx.filter_vmap(loss_fn, in_axes=(None, 0, 0))


# Example of how you would use it in a training loop
def batch_loss(
    model: DiffusionModel,
    x_0_batch: Float[Array, "batch channel height width"],
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    batch_size = x_0_batch.shape[0]
    keys = jax.random.split(key, num=batch_size)

    # Calculate loss for each image in the batch
    losses = batched_loss_fn(model, x_0_batch, keys)

    # Return the average loss over the batch
    return jnp.mean(losses)


# 2. Create the jitted training step function
@eqx.filter_jit
def train_step(
    model: DiffusionModel,
    opt_state: optax.OptState,
    x_0_batch: Float[Array, "batch channel height width"],
    key: PRNGKeyArray,
):
    loss_value, grads = eqx.filter_value_and_grad(batch_loss)(model, x_0_batch, key)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


num_epochs = 10
batch_size = 32
learning_rate = 1e-4
time_embedding_size = 32
timesteps = 200

main_key = jax.random.key(42)
model_key, train_key = jax.random.split(main_key)

model = DiffusionModel(time_embedding_size, key=model_key)
optimizer = optax.adam(learning_rate)


opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

betas = linear_beta_schedule(timesteps=timesteps)
alphas = 1.0 - betas
alpha_bars = jnp.cumprod(alphas, axis=0)
sqrt_alpha_bars = jnp.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = jnp.sqrt(1.0 - alpha_bars)


train_data = load_mnist_from_hf("train")
test_data = load_mnist_from_hf("test")

train_source = DataFrameDataSource(train_data)
test_source = DataFrameDataSource(test_data)

transformations = [MNISTPreprocessor(), grain.transforms.Batch(batch_size=batch_size)]


index_sampler = grain.samplers.IndexSampler(
    num_records=len(train_source),
    num_epochs=2,
    shard_options=grain.sharding.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True
    ),
    shuffle=True,
    seed=0,
)

data_loader = grain.DataLoader(
    data_source=train_source,  # pyright: ignore
    operations=transformations,
    sampler=index_sampler,
    worker_count=0,
)

print(summarize_model(model))

# for epoch in range(num_epochs):
#     step = 0
#     total_loss = 0
#     with tqdm(data_loader, unit="batch") as tepoch:
#         tepoch.set_description(f"Epoch {epoch + 1}")
#         for batch in tepoch:
#             x_0_batch = batch["image"]
#             train_key, step_key = jax.random.split(train_key)
#             model, opt_state, loss = train_step(model, opt_state, x_0_batch, step_key)

#             total_loss += loss.item()
#             step += 1

#             if step % 10 == 0:
#                 avg_loss = total_loss / 100
#                 tepoch.set_postfix(loss=avg_loss)
#                 total_loss = 0
