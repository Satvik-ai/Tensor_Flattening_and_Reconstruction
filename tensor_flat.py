import numpy as np


# ---------------------------------------------------------
# BCHW -> B x (CHW)
# ---------------------------------------------------------
def flatten_bchw(tensor):
    B, C, H, W = tensor.shape

    # Create output tensor of shape B x (CHW)
    flattened = np.empty((B, C * H * W), dtype=tensor.dtype)

    # Explicit indexing
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):

                    # Calculate flattened index
                    k = c * H * W + h * W + w

                    # Copy value
                    flattened[b, k] = tensor[b, c, h, w]

    return flattened


# ---------------------------------------------------------
# B x (CHW) -> BCHW
# ---------------------------------------------------------
def reconstruct_bchw(flattened, B, C, H, W):

    # Create output tensor of shape B x C x H x W
    reconstructed = np.empty(
        (B, C, H, W),
        dtype=flattened.dtype
    )

    # Explicit indexing
    for b in range(B):
        for k in range(C * H * W):

            # Recover channel
            c = k // (H * W)

            # Remaining index after removing channel
            remaining = k % (H * W)

            # Recover height
            h = remaining // W

            # Recover width
            w = remaining % W

            # Copy value back
            reconstructed[b, c, h, w] = flattened[b, k]

    return reconstructed


# ---------------------------------------------------------
# Calculate reconstruction errors
# ---------------------------------------------------------
def calculate_errors(original, reconstructed):

    # Element-wise error
    error = original - reconstructed

    # Maximum absolute error
    emax = np.max(np.abs(error))

    # Mean absolute error
    mae = np.mean(np.abs(error))

    return emax, mae


# ---------------------------------------------------------
# Test one tensor configuration
# ---------------------------------------------------------
def run_experiment(B, C, H, W):

    print("\n----------------------------------------")
    print(f"B = {B}, C = {C}, H = {H}, W = {W}")
    print("----------------------------------------")

    # Generate synthetic tensor
    tensor = np.random.rand(B, C, H, W)

    # BCHW -> B x (CHW)
    flattened = flatten_bchw(tensor)

    # B x (CHW) -> BCHW
    reconstructed = reconstruct_bchw(
        flattened,
        B,
        C,
        H,
        W
    )

    # Calculate errors
    emax, mae = calculate_errors(
        tensor,
        reconstructed
    )

    print("Original shape      :", tensor.shape)
    print("Flattened shape     :", flattened.shape)
    print("Reconstructed shape :", reconstructed.shape)

    print("Maximum absolute error :", emax)
    print("Mean absolute error    :", mae)

    return emax, mae


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
if __name__ == "__main__":

    # Fixed B, H and W
    B = 2
    H = 8
    W = 8

    # Different channel counts
    channel_counts = [1, 3, 8, 16, 32, 64, 128, 256, 500]

    results = []

    for C in channel_counts:

        emax, mae = run_experiment(
            B,
            C,
            H,
            W
        )

        results.append(
            (B, C, H, W, emax, mae)
        )

    # -----------------------------------------------------
    # Display final results table
    # -----------------------------------------------------

    print("\n\n================ EXPERIMENTAL RESULTS ================")

    print(
        f"{'B':<6}"
        f"{'C':<8}"
        f"{'H':<8}"
        f"{'W':<8}"
        f"{'Emax':<20}"
        f"{'MAE':<20}"
    )

    print("-" * 70)

    for B, C, H, W, emax, mae in results:

        print(
            f"{B:<6}"
            f"{C:<8}"
            f"{H:<8}"
            f"{W:<8}"
            f"{emax:<20.10e}"
            f"{mae:<20.10e}"
        )