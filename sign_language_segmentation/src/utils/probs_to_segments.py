import numpy as np
import os
import sys

BIO = {"O": 0, "B": 1, "I": 2}


def io_probs_to_segments(probs):
    segments = []
    i = 0
    while i < len(probs):
        if probs[i, BIO["I"]] > 50:
            end = len(probs) - 1
            for j in range(i + 1, len(probs)):
                if probs[j, BIO["I"]] < 50:
                    end = j - 1
                    break
            segments.append({"start": i, "end": end})
            i = end + 1
        else:
            i += 1

    return segments


def probs_to_segments(logits, b_threshold=50., o_threshold=50., threshold_likeliest=False, restart_on_b=True):
    probs = np.round(np.exp(logits.numpy().squeeze()) * 100)

    np.set_printoptions(threshold=np.inf)

    print(probs[:, BIO["B"]])

    if np.alltrue(probs[:, BIO["B"]] < b_threshold):
        print(f"b_threshold:\t{str(b_threshold)}")
        print('np.alltrue(probs[:, BIO["B"]] < b_threshold')
        # sys.exit(1)
        return io_probs_to_segments(probs)


    print(f"probs shape:\t{str(probs.shape)}")
    print(f"\nprobs:\n{str(probs)}\n")

    with open(os.path.join("/home/ec2-user/andrew_messaround/vsl_phrase_segmentation/messaround", "example_numpy_tensor_right.txt"), "w", encoding="utf-8") as fw:
        print("Saving numpy tensor of probabilities to disk.")
        fw.write(str(probs))
        print("…Saved.")

    segments = []

    segment = {"start": None, "end": None}
    did_pass_start = False
    for idx in range(len(probs)):
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        if threshold_likeliest:
            b_threshold = max(i, o)
            o_threshold = max(b, i)

        if segment["start"] is None:
            if b > b_threshold:
                segment["start"] = idx
        else:
            if did_pass_start:
                if (restart_on_b and b > b_threshold) or o > o_threshold:
                    segment["end"] = idx - 1

                    # reset
                    segments.append(segment)
                    segment = {"start": None if o > o_threshold else idx, "end": None}
                    did_pass_start = False
            else:
                if b < b_threshold:
                    did_pass_start = True

    if segment["start"] is not None:
        segment["end"] = len(probs)
        segments.append(segment)

    return segments


def custom_probs_to_segments(probs, b_threshold, i_threshold, o_threshold):
    # Convert logits to probabilities and scale them
    # probs = np.round(np.exp(logits.numpy().squeeze()) * 100)

    np.set_printoptions(threshold=np.inf)

    segments = []
    segment = {"start": None, "end": None}

    idx = 0
    while idx < len(probs):
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        if b >= b_threshold or i >= i_threshold:
            if o < o_threshold:
                if segment["start"] is None:
                    segment["start"] = idx  # Start a new segment
            elif o >= o_threshold:
                if segment["start"] is not None:
                    segment["end"] = idx - 1  # End the current segment
                    if segment["end"] > segment["start"]:  # Ensure segment length is at least one frame
                        segments.append(segment)
                    segment = {"start": None, "end": None}  # Reset the segment
        else:
            if segment["start"] is not None:
                segment["end"] = idx - 1  # End the current segment
                if segment["end"] > segment["start"]:  # Ensure segment length is at least one frame
                    segments.append(segment)
                segment = {"start": None, "end": None}  # Reset the segment

        idx += 1

    # Handle the case where the last segment reaches the end of the array
    if segment["start"] is not None:
        segment["end"] = len(probs) - 1
        if segment["end"] > segment["start"]:  # Ensure segment length is at least one frame
            segments.append(segment)

    return segments




